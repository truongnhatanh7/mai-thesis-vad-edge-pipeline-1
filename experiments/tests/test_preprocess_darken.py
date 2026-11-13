"""Guards the synthetic low-light model: identity at alpha=1, monotone darkening, reproducible noise."""

import numpy as np
import pytest

from edgelab.preprocess.darken import (LEVELS, darken, darken_level, level_seed, linear_to_srgb, mean_luma,
                                       srgb_to_linear)


def _image(seed=0, shape=(48, 64, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _smooth_image():
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)
    return np.tile(ramp[None, :, None], (48, 1, 3))


def test_srgb_roundtrip_and_known_points():
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-9)
    assert srgb_to_linear(np.array(0.5)) == pytest.approx(0.21404, abs=1e-4)  # well-known mid-grey value


def test_alpha_one_without_noise_is_identity_within_one_level():
    img = _image()
    out = darken(img, alpha=1.0, noise=False)
    assert out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_mean_luma_decreases_with_alpha():
    img = _smooth_image()
    lumas = [mean_luma(darken(img, a, seed=1)) for a in (1.0, 0.3, 0.1, 0.03, 0.01)]
    assert all(a > b for a, b in zip(lumas, lumas[1:]))


def test_same_seed_same_noise_different_seed_different_noise():
    img = _image()
    np.testing.assert_array_equal(darken(img, 0.1, seed=5), darken(img, 0.1, seed=5))
    assert not np.array_equal(darken(img, 0.1, seed=5), darken(img, 0.1, seed=6))


def test_without_noise_output_is_deterministic_and_smooth():
    img = _smooth_image()
    a, b = darken(img, 0.1, noise=False, seed=1), darken(img, 0.1, noise=False, seed=2)
    np.testing.assert_array_equal(a, b)
    row = a[0, :, 0].astype(int)
    assert np.all(np.diff(row) >= 0)  # a monotone ramp stays monotone: no noise


def test_noise_makes_a_flat_dark_patch_vary_and_averages_near_noiseless():
    flat = np.full((100, 100, 3), 120, np.uint8)
    noisy, clean = darken(flat, 0.1, seed=3), darken(flat, 0.1, noise=False)
    assert noisy.std() > 1.0 and clean.std() == 0.0
    assert abs(noisy.mean() - clean.mean()) < 3.0


def test_level_seed_is_stable_and_distinguishes_inputs():
    assert level_seed("01_0014", 7, "L2") == level_seed("01_0014", 7, "L2")
    assert len({level_seed("01_0014", 7, "L2"), level_seed("01_0014", 8, "L2"),
                level_seed("01_0015", 7, "L2"), level_seed("01_0014", 7, "L3")}) == 4


def test_level_seed_value_does_not_change_between_runs():
    # Pinned: if this changes, every cached dark frame changes (the "stable hash" promise is broken).
    assert level_seed("01_0014", 7, "L2") == 7397195211593958909


def test_levels_table_matches_plan():
    plan = {"L0": None, "L1": 0.3, "L2": 0.1, "L3": 0.03, "L4": 0.01, "L2-nonoise": 0.1}
    addendum_b = {"L2-s1": 0.1, "L3-s1": 0.03}  # second noise realisation (plan §8)
    exp005_held_out = {"X1": 0.2, "X2": 0.05, "X3": 0.02}
    assert LEVELS == {**plan, **addendum_b, **exp005_held_out}


def test_darken_level_l0_is_untouched_and_nonoise_ignores_seed():
    img = _image()
    assert darken_level(img, "L0", "v", 0) is img
    np.testing.assert_array_equal(darken_level(img, "L2-nonoise", "v", 0), darken_level(img, "L2-nonoise", "w", 9))
    assert not np.array_equal(darken_level(img, "L2", "v", 0), darken_level(img, "L2", "v", 1))


def test_gain_level_uses_base_noise_and_only_scales_signal():
    from edgelab.preprocess.darken import darken_level, level_seed, darken, LEVELS
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    base = darken_level(img, "L2", "01_0014", 5)
    assert np.array_equal(base, darken(img, LEVELS["L2"], seed=level_seed("01_0014", 5, "L2"), gain=1.0))
    brighter = darken_level(img, "L2-gain", "01_0014", 5)
    assert brighter.mean() > base.mean() + 30
    # same noise realisation: a pixel that is darker than another before gain is never brighter after it
    b, g = base.ravel().astype(int), brighter.ravel().astype(int)
    i, j = rng.integers(0, b.size, 5000), rng.integers(0, b.size, 5000)
    assert not np.any((b[i] < b[j]) & (g[i] > g[j]))
