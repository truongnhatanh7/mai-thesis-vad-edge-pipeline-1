"""EXP-019 / EXP-018 flow: shapes, invertibility, log-determinant, equivalence with third_party STG-NF, GAU masking."""

import numpy as np
import pytest
import torch

from edgelab.models.stgnf_flow import (CouplingFlow, GatedAttentionUnit, build_flow, count_parameters,
                                       make_adjacency, random_pose_batch)
from edgelab.pose.stgnf_eval import _import_stgnf

_import_stgnf()
from models.STG_NF.model_pose import STG_NF  # noqa: E402


def initialised(model: CouplingFlow, x: torch.Tensor, provenance=None) -> CouplingFlow:
    """One training-mode pass sets ActNorm from the data (as the first training batch does); then eval mode."""
    for p in model.parameters():  # make every weight non-zero so the subnets really do something
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.05)
    model.train()
    model.encode(x, provenance)
    return model.eval()


def third_party_model(steps: int, n_frames: int = 24) -> STG_NF:
    return STG_NF(pose_shape=(2, n_frames, 18), hidden_channels=0, K=steps, L=1, R=3.0, actnorm_scale=1.0,
                  flow_permutation="permute", flow_coupling="affine", LU_decomposed=True, learn_top=False,
                  edge_importance=False, temporal_kernel_size=None, strategy="uniform", max_hops=8, device="cpu")


def test_graph_flow_has_the_parameters_of_the_original_and_gau_stays_under_ten_times():
    original = count_parameters(third_party_model(8))
    assert count_parameters(build_flow("graph")) == original == 616
    assert count_parameters(build_flow("gau")) <= 10 * original
    assert count_parameters(build_flow("gau-fa")) <= 10 * original


def test_graph_flow_equals_third_party_flow_with_the_same_weights():
    torch.manual_seed(0)
    theirs, ours = third_party_model(3), build_flow("graph")
    ours.steps = ours.steps[:3]
    x = random_pose_batch(16, seed=1)
    theirs.train()
    theirs(x, label=torch.ones(16), score=1)  # ActNorm init on this batch
    for mine, step in zip(ours.steps, theirs.flow.layers):
        mine.actnorm.load_state_dict(step.actnorm.state_dict())
        mine.subnet.gcn.load_state_dict(step.block[0].state_dict())
        mine.actnorm.inited = True
    for p in list(theirs.parameters()):  # non-trivial subnets: perturb theirs, copy again
        p.data += 0.05 * torch.randn_like(p)
    for mine, step in zip(ours.steps, theirs.flow.layers):
        mine.actnorm.load_state_dict(step.actnorm.state_dict())
        mine.subnet.gcn.load_state_dict(step.block[0].state_dict())
    theirs.eval(), ours.eval()
    with torch.no_grad():
        z_theirs, nll_theirs = theirs(x, label=torch.ones(16), score=1)
        z_ours, nll_ours = ours(x)
    assert torch.allclose(z_ours, z_theirs, atol=1e-5)
    assert torch.allclose(nll_ours, nll_theirs, atol=1e-5)


@pytest.mark.parametrize("kind", ["graph", "gau", "graph-fa", "gau-fa", "gau-fa-mask"])
def test_flow_keeps_shape_and_is_invertible(kind):
    torch.manual_seed(0)
    x = random_pose_batch(8, seed=2)
    provenance = (torch.rand(8, 24) > 0.4).float() if kind.endswith(("fa", "mask")) else None
    model = initialised(build_flow(kind), x, provenance)
    with torch.no_grad():
        z, nll = model(x, provenance)
        back = model.decode(z, provenance)
    assert z.shape == x.shape and nll.shape == (8,)
    assert torch.allclose(back, x, atol=1e-4)


@pytest.mark.parametrize("kind", ["gau", "gau-fa-mask"])
def test_log_determinant_equals_the_jacobian(kind):
    """T = 8 keeps the Jacobian small (288 x 288); logdet from the flow == slogdet of the autograd Jacobian."""
    torch.manual_seed(1)
    x = random_pose_batch(4, n_frames=8, seed=3)
    provenance = (torch.rand(1, 8) > 0.3).float() if kind == "gau-fa-mask" else None
    model = build_flow(kind, n_frames=8)
    model.steps = model.steps[:2]
    initialised(model, x, None if provenance is None else provenance.expand(4, 8))
    sample = x[:1]
    with torch.no_grad():
        _, logdet = model.encode(sample, provenance)
    jacobian = torch.autograd.functional.jacobian(lambda v: model.encode(v.reshape(1, 2, 8, 18), provenance)[0].reshape(-1),
                                                  sample.reshape(-1))
    assert logdet.item() == pytest.approx(torch.linalg.slogdet(jacobian)[1].item(), abs=1e-3)


def test_gau_adds_nothing_at_initialisation_and_graph_part_is_the_original():
    torch.manual_seed(0)
    x = random_pose_batch(4, seed=4)
    gau, graph = build_flow("gau"), build_flow("graph")
    for a, b in zip(gau.steps, graph.steps):
        b.subnet.gcn.load_state_dict(a.subnet.graph.gcn.state_dict())
    gau.train(), graph.train()
    z_gau, nll_gau = gau(x)
    z_graph, nll_graph = graph(x)
    assert torch.allclose(nll_gau, nll_graph, atol=1e-5) and torch.allclose(z_gau, z_graph, atol=1e-5)


def test_provenance_changes_the_density_and_a_flow_without_it_rejects_none():
    torch.manual_seed(0)
    x = random_pose_batch(4, seed=5)
    all_detected, half_predicted = torch.ones(4, 24), torch.ones(4, 24)
    half_predicted[:, 12:] = 0
    model = initialised(build_flow("graph-fa"), x, all_detected)
    with torch.no_grad():
        assert not torch.allclose(model(x, all_detected)[1], model(x, half_predicted)[1])
    with pytest.raises(ValueError):
        model(x, None)


def test_gau_masking_ignores_missing_keys_only():
    torch.manual_seed(0)
    gau = GatedAttentionUnit(in_channels=2, n_frames=10, mask_missing=True)
    torch.nn.init.normal_(gau.to_out.weight, std=0.3)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[:, 4] = False  # frame 4 is missing
    x = torch.randn(2, 2, 10, 18)
    changed = x.clone()
    changed[:, :, 4] += 5.0  # whatever sits in the missing frame
    with torch.no_grad():
        a, b = gau(x, valid), gau(changed, valid)
        unmasked_a, unmasked_b = gau(x, None), gau(changed, None)
    others = [t for t in range(10) if t != 4]
    assert torch.allclose(a[:, :, others], b[:, :, others], atol=1e-6)  # other frames cannot see it
    assert not torch.allclose(a[:, :, 4], b[:, :, 4])  # its own query still changes
    assert not torch.allclose(unmasked_a[:, :, others], unmasked_b[:, :, others], atol=1e-6)  # without the mask they do


def test_adjacency_is_the_released_one():
    assert make_adjacency().shape == (1, 18, 18)
    assert np.allclose(make_adjacency()[0].sum(0), 1.0)  # column-normalised


def test_relative_bias_matrix_is_toeplitz_in_t_minus_t_prime():
    gau = GatedAttentionUnit(in_channels=1, n_frames=5)
    with torch.no_grad():
        gau.rel_bias.copy_(torch.arange(9.0))  # bias value = its offset index: [t, t'] -> t - t' + 4
    matrix = gau.bias_matrix()
    assert all(matrix[t, u] == t - u + 4 for t in range(5) for u in range(5))
