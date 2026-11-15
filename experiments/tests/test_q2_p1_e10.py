"""Unit tests for Q2-P1-E10 Fast-Feedback Contextual Bandit / RL OS Tuner (`FastOS-RL`)."""

from __future__ import annotations

import numpy as np

from edgelab.analysis.q2_p1_e10 import estimate_delivered_auc_from_pi5_service
from edgelab.analysis.q2_p1_e10_pi5 import (
    FastOSLinUCB,
    build_action_space,
    compute_rl_reward,
)


def test_action_space_and_linucb_convergence():
    actions = build_action_space()
    assert len(actions) == 18  # 3 governors x 3 thread counts x 2 pinning modes

    bandit = FastOSLinUCB(actions, alpha=0.25, warm_start=True)
    ctx_bursty = {"is_cached_bursty": 1.0, "res_scale": 1.0}

    # Simulate synthetic environment where performance governor + pinned T=3/T=4 gives highest reward on bursty workload
    for step in range(20):
        a, _, _ = bandit.select_action(ctx_bursty, step)
        synth_reward = -1.0
        if a.governor == "performance":
            synth_reward += 0.25
        if a.pin_isolated:
            synth_reward += 0.20
        if a.threads >= 3:
            synth_reward += 0.15
        bandit.update(ctx_bursty, a, synth_reward)

    best = bandit.best_greedy_action(ctx_bursty)
    assert best.governor == "performance"
    assert best.pin_isolated is True
    assert best.threads in (3, 4)


def test_reward_and_delivered_auc_estimator():
    m_fast = {
        "drop_rate_pct": 2.0,
        "service_ms_mean": 30.0,
        "service_ms_p95": 38.0,
        "energy_j_per_frame": 0.18,
    }
    m_slow = {
        "drop_rate_pct": 45.0,
        "service_ms_mean": 58.0,
        "service_ms_p95": 72.0,
        "energy_j_per_frame": 0.32,
    }
    assert compute_rl_reward(m_fast) > compute_rl_reward(m_slow)

    auc_fast = estimate_delivered_auc_from_pi5_service(76.82, 31.0, 2.0, use_cache_and_guard=True)
    auc_slow_unguarded = estimate_delivered_auc_from_pi5_service(76.82, 60.0, 80.0, use_cache_and_guard=False)
    assert auc_fast > 74.0
    assert auc_slow_unguarded < 59.0
