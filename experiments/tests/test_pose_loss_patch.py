"""EXP-020: the keypoint-loss patch. Stock behaviour for keypoint-less persons is recorded, patched behaviour is box-only."""

from types import SimpleNamespace

import torch
from torch import nn
from ultralytics.utils.loss import KeypointLoss
from ultralytics.utils.metrics import OKS_SIGMA

from edgelab.pose.loss_patch import STOCK, install, masked_calculate_keypoints_loss
from ultralytics.utils.loss import v8PoseLoss


def _loss_object():
    return SimpleNamespace(keypoint_loss=KeypointLoss(torch.from_numpy(OKS_SIGMA)), bce_pose=nn.BCEWithLogitsLoss())


def _batch(person_has_keypoints: list[bool], seed: int = 0):
    """One image, one positive anchor per person (anchor i -> person i). Returns the args of calculate_keypoints_loss."""
    g = torch.Generator().manual_seed(seed)
    n = len(person_has_keypoints)
    masks = torch.ones(1, n, dtype=torch.bool)
    target_gt_idx = torch.arange(n).view(1, n)
    keypoints = torch.rand(1, n, 17, 3, generator=g) * 100  # pixels
    keypoints[..., 2] = 2.0
    for i, has in enumerate(person_has_keypoints):
        if not has:
            keypoints[0, i] = 0.0  # x = y = v = 0 for all 17
    stride = torch.full((n, 1), 8.0)
    boxes = torch.tensor([[10.0, 10.0, 90.0 + 5 * i, 100.0 + 5 * i] for i in range(n)]).view(1, n, 4)
    pred = (torch.rand(1, n, 17, 3, generator=g) * 4).requires_grad_(True)
    return masks, target_gt_idx, keypoints, stride, boxes, pred


def _call(fn, args):
    masks, idx, kp, stride, boxes, pred = args
    return fn(_loss_object(), masks, idx, kp.clone(), stride, boxes.clone(), pred)


def test_patched_equals_stock_when_every_person_has_keypoints():
    args = _batch([True, True, True])
    stock, patched = _call(STOCK, args), _call(masked_calculate_keypoints_loss, args)
    assert torch.equal(stock[0], patched[0]) and torch.equal(stock[1], patched[1])


def _subset(args, keep: list[int]):
    """Same batch with only the anchors `keep` positive; target_gt_idx still points into the full keypoint tensor."""
    masks, idx, kp, stride, boxes, pred = args
    k = torch.tensor(keep)
    return masks[:, k], idx[:, k], kp, stride[k], boxes[:, k], pred[:, k]


def test_stock_loss_for_keypoint_less_person_trains_visibility_and_dilutes_location():
    """Documents the stock behaviour that motivates the patch."""
    mixed = _batch([True, True, False])
    stock_mixed = _call(STOCK, mixed)
    stock_only_kp = _call(STOCK, _subset(mixed, [0, 1]))
    assert not torch.isclose(stock_mixed[0], stock_only_kp[0])  # location loss diluted by the keypoint-less anchor
    assert not torch.isclose(stock_mixed[1], stock_only_kp[1])  # visibility BCE also sees the keypoint-less person (target 0)


def test_patched_mixed_batch_equals_stock_on_keypoint_persons_only():
    mixed = _batch([True, False, True, False])
    patched = _call(masked_calculate_keypoints_loss, mixed)
    stock = _call(STOCK, _subset(mixed, [0, 2]))
    assert torch.allclose(stock[0], patched[0]) and torch.allclose(stock[1], patched[1])


def test_patched_all_keypoint_less_is_zero_and_backpropagates():
    args = _batch([False, False])
    loc, vis = _call(masked_calculate_keypoints_loss, args)
    assert float(loc) == 0.0 and float(vis) == 0.0
    (loc + vis).backward()  # graph stays connected (no error), gradient is zero
    assert args[5].grad is not None and float(args[5].grad.abs().sum()) == 0.0


def test_gradient_of_patched_loss_ignores_keypoint_less_anchors():
    args = _batch([True, False, True])
    loc, vis = _call(masked_calculate_keypoints_loss, args)
    (loc + vis).backward()
    grad = args[5].grad
    assert float(grad[0, 1].abs().sum()) == 0.0 and float(grad[0, 0].abs().sum()) > 0.0


def test_install_replaces_the_class_method_and_stock_is_restorable():
    original = v8PoseLoss.calculate_keypoints_loss
    try:
        install()
        assert v8PoseLoss.calculate_keypoints_loss is masked_calculate_keypoints_loss
    finally:
        v8PoseLoss.calculate_keypoints_loss = original
