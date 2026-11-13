"""EXP-020: keypoint losses only for persons that have keypoint labels (partial-label training).

Ultralytics 8.4.201 `v8PoseLoss.calculate_keypoints_loss` (utils/loss.py) takes, for every positive anchor, the GT person's
keypoints and builds `kpt_mask = (visibility != 0)`. For a person written with 17 keypoints at (0, 0, 0):
  * the keypoint LOCATION loss is exactly zero for that anchor (`... * kpt_mask`), but the anchor still counts in the
    `.mean()` over anchors, which dilutes the location loss of all other persons;
  * the keypoint VISIBILITY loss (BCE of the predicted visibility against `kpt_mask`) is NOT masked: it trains the head to
    say "all 17 keypoints invisible" for that person.
The plan says keypoint-less persons are box-only targets, so this patch drops the anchors whose GT person has no visible
keypoint from BOTH keypoint terms. The box / class / DFL losses are untouched (they are computed elsewhere).
For a batch in which every GT person has a keypoint the patched function equals the stock one (tests/test_pose_loss_patch.py),
so it is installed for all four EXP-020 runs.
"""

from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.ops import xyxy2xywh

STOCK = v8PoseLoss.calculate_keypoints_loss  # kept for the equality test


def masked_calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, stride_tensor, target_bboxes, pred_kpts):
    """Stock body (8.4.201) plus: positives whose GT person has no visible keypoint are removed before both losses."""
    kpts_loss = 0
    kpts_obj_loss = 0
    if masks.any():
        target_bboxes /= stride_tensor
        gt_kpt = keypoints[masks.nonzero(as_tuple=True)[0], target_gt_idx[masks]]
        gt_kpt[..., :2] /= stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks][:, None, None]
        area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
        pred_kpt = pred_kpts[masks]
        kpt_mask = gt_kpt[..., 2] != 0
        has_keypoints = kpt_mask.any(1)  # one entry per positive anchor
        if not has_keypoints.any():  # only keypoint-less persons assigned: no keypoint loss, graph stays connected
            zero = pred_kpt.sum() * 0.0
            return zero, zero
        gt_kpt, area, pred_kpt, kpt_mask = gt_kpt[has_keypoints], area[has_keypoints], pred_kpt[has_keypoints], kpt_mask[has_keypoints]
        kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
        if pred_kpt.shape[-1] == 3:
            kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
    return kpts_loss, kpts_obj_loss


def install():
    """Replace the method on the class (affects every v8PoseLoss created afterwards in this process)."""
    v8PoseLoss.calculate_keypoints_loss = masked_calculate_keypoints_loss

