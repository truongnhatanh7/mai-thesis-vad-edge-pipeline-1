"""STEP 3: Offline (`S0`) vs. Strict Causal (`S4`) Evaluation on ShanghaiTech (107 Clips).

Read this file third (~170 lines). It shows how per-window normality scores are turned into frame-level ROC-AUC,
and exposes the exact difference between:
  - `S0` (Offline Paper Protocol used by STG-NF, Huang et al., and LoRaD):
    1. Lands each 24-frame window's score at the MIDDLE frame (`start + 12`, looking 12 frames into the future).
    2. Fills frames with no person using `max(all_test_scores)` (a look-ahead statistic over the whole test set).
    3. Smooths with a two-sided Gaussian filter (`sigma = 1..6`, averaging past AND future frames).
  - `S4` (Strict Causal Real-Time Protocol from Q2-P1-E00):
    1. Lands each window's score at its actual COMPLETION frame (`last_frame` of the 24th pose).
    2. Fills frames with no person using a fixed sentinel calibrated ONLY on normal training videos (`-0.5122`).
    3. Smooths with a one-sided causal half-Gaussian filter (using past frames `<= t` only).

Run directly:
    cd /Users/nanh/Documents/RMIT_MscAI/thesis/option_a_clean_review
    ../experiments/.venv/bin/python step3_offline_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "experiments" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "src"))

from step1_models import ema_normalize_track_n1, load_scorer_checkpoint, normalize_window_n0  # noqa: E402


# =====================================================================
# PART A: Temporal Smoothers (Two-Sided Offline vs. One-Sided Causal)
# =====================================================================
def smooth_offline_two_sided(scores: np.ndarray) -> np.ndarray:
    """STG-NF's original offline smoother: applies two-sided Gaussian filters sigma=1..6 sequentially (uses future!)."""
    out = np.asarray(scores, dtype=np.float64).copy()
    for sigma in range(1, 7):
        out = gaussian_filter1d(out, sigma=sigma)
    return out


def smooth_causal_one_sided(scores: np.ndarray) -> np.ndarray:
    """Strict causal half-Gaussian smoother (sigma = sqrt(sum_{s=1..6} s^2) = 9.539): uses ONLY past frames <= t."""
    sigma = float(np.sqrt(sum(s * s for s in range(1, 7))))
    radius = int(np.ceil(4.0 * sigma))
    lags = np.arange(radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (lags / sigma) ** 2)
    kernel /= kernel.sum()
    x = np.asarray(scores, dtype=np.float64)
    padded = np.pad(x, (radius, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# =====================================================================
# PART B: Score Assembly (S0 Offline Paper vs. S4 Strict Causal)
# =====================================================================
def evaluate_tracks_s0_and_s4(
    tracks_by_clip: list[dict[int, tuple[np.ndarray, np.ndarray]]],
    gt_labels_by_clip: list[np.ndarray],
    flow_model: torch.nn.Module,
    seg_len: int = 24,
    norm_mode: str = "N0",
    train_sentinel: float = -0.5122,
) -> dict[str, float]:
    """Score all 24-frame windows across all clips and compute both S0 (offline paper) and S4 (strict causal) AUC."""
    flow_model.eval()
    windows_list = []
    meta = []  # (clip_idx, start_frame, completion_frame, is_not_last_window)

    for clip_idx, trk_dict in enumerate(tracks_by_clip):
        for tid, (frames, kp) in trk_dict.items():
            n = len(frames)
            if n < seg_len:
                continue
            # Find all valid 24-pose windows with <= 2 missing frames
            s_indices = np.arange(n - seg_len + 1)
            in_span = np.searchsorted(frames, frames[s_indices] + seg_len, side="left") - s_indices
            valid_starts = s_indices[np.minimum(in_span, seg_len) >= seg_len - 2]
            if len(valid_starts) == 0:
                continue

            idx = valid_starts[:, None] + np.arange(seg_len)[None, :]
            if norm_mode == "N0":
                windows_list.append(kp[:, :17, :2][idx])  # (M, 24, 17, 2) raw pixels
            else:
                # N1 streaming prefix normalization (computed once per pose as it arrives)
                normed_track = ema_normalize_track_n1(kp[:, :17, :2], seg_len=seg_len)
                windows_list.append(normed_track[idx])  # (M, 24, 18, 2) pre-normalized

            # Note: original offline STG-NF had a ceil((n-24)/1) bug that dropped the very last window of each track
            offline_limit = int(np.ceil((n - seg_len) / 1.0))
            for st in valid_starts:
                meta.append((clip_idx, int(frames[st]), int(frames[st + seg_len - 1]), bool(st < offline_limit)))

    if not windows_list:
        return {"auc_s0": 50.0, "auc_s4": 50.0, "coverage_pct": 0.0}

    all_wins = np.concatenate(windows_list, axis=0)
    if norm_mode == "N0":
        x_tensor = normalize_window_n0(all_wins)
    else:
        x_tensor = torch.from_numpy(np.ascontiguousarray(all_wins.transpose(0, 3, 1, 2), dtype=np.float32))

    normality_parts = []
    with torch.no_grad():
        for i in range(0, len(x_tensor), 2048):
            batch = x_tensor[i : i + 2048]
            try:
                _, nll = flow_model(batch, label=torch.ones(len(batch)))
            except TypeError:
                _, nll = flow_model(batch)
            normality_parts.append((-nll).cpu().numpy().reshape(-1))
    normality = np.concatenate(normality_parts, axis=0)

    # Assemble per-clip frame normality arrays
    s0_clips = [np.full(len(gt), np.nan, dtype=np.float64) for gt in gt_labels_by_clip]
    s4_clips = [np.full(len(gt), np.nan, dtype=np.float64) for gt in gt_labels_by_clip]
    mid_offset = seg_len // 2

    for val, (c_idx, st_f, end_f, in_off) in zip(normality, meta):
        n_f = len(s0_clips[c_idx])
        # S0: middle frame landing (only if not omitted by STG-NF's last-window bug)
        if in_off:
            mf = min(n_f - 1, st_f + mid_offset)
            if np.isnan(s0_clips[c_idx][mf]) or val < s0_clips[c_idx][mf]:
                s0_clips[c_idx][mf] = float(val)
        # S4: strict completion frame landing (all complete windows)
        ef = min(n_f - 1, end_f)
        if np.isnan(s4_clips[c_idx][ef]) or val < s4_clips[c_idx][ef]:
            s4_clips[c_idx][ef] = float(val)

    # S0 fills empty frames with test-set global max; S4 uses strict training-only sentinel
    test_global_max = float(np.nanmax(normality))
    s0_scores = [smooth_offline_two_sided(-np.where(np.isfinite(c), c, test_global_max)) for c in s0_clips]
    s4_scores = [smooth_causal_one_sided(-np.where(np.isfinite(c), c, train_sentinel)) for c in s4_clips]

    y_all = np.concatenate(gt_labels_by_clip)
    cov_pct = 100.0 * float(np.mean(np.isfinite(np.concatenate(s4_clips))))
    return {
        "auc_s0": round(100.0 * float(roc_auc_score(y_all, np.concatenate(s0_scores))), 2),
        "auc_s4": round(100.0 * float(roc_auc_score(y_all, np.concatenate(s4_scores))), 2),
        "coverage_pct": round(cov_pct, 1),
    }


def main() -> None:
    """Run the clean Option A offline/causal comparison across Huang et al., LoRaD, and Ours on ShanghaiTech (107 clips)."""
    import os
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    torch.set_num_threads(1)
    from edgelab.analysis import q2_p1_e00 as e00
    from edgelab.analysis import q2_p1_e06 as e06
    from edgelab.analysis import q2_p1_eval as ev

    ctx = ev.load_ctx()["ShanghaiTech"]
    stgnf_ckpt = REPO_ROOT / "experiments/third_party/STG-NF/checkpoints/ShanghaiTech_85_9.tar"
    lorad_ckpt = Path("~/thesis-data/q2-p1-e06/models/A2_ShanghaiTech_s0/final.pt").expanduser()
    lorad_n1_ckpt = Path("~/thesis-data/q2-p1-e06/models/A3_ShanghaiTech_s0/final.pt").expanduser()

    stgnf_model = load_scorer_checkpoint(stgnf_ckpt)
    lorad_n0_model = load_scorer_checkpoint(lorad_ckpt)
    lorad_n1_model = load_scorer_checkpoint(lorad_n1_ckpt)

    print("\n=== Option A: ShanghaiTech Full 107-Clip Offline (S0) vs. Strict Causal (S4) Audit ===", flush=True)
    print(f"{'Pipeline Configuration':52s} | {'Offline S0 AUC':>14s} | {'Causal S4 AUC':>13s} | {'Scored Cov':>10s}", flush=True)
    print("-" * 98, flush=True)

    # 1. AlphaPose Oracle (what Huang et al. 86.2% & LoRaD 84.8% actually benchmarked on)
    trks_released = e06.pose_tracks("ShanghaiTech", "released", ctx, procs=2)
    for label, mdl, nmode in [
        ("1a. Oracle AlphaPose + STG-NF (Paper Reference)", stgnf_model, "N0"),
        ("1b. Oracle AlphaPose + LoRaD Causal Flow (N0 Window)", lorad_n0_model, "N0"),
        ("1c. Oracle AlphaPose + LoRaD Prefix-Norm (N1 Stream)", lorad_n1_model, "N1"),
    ]:
        r = evaluate_tracks_s0_and_s4(trks_released, ctx.labels, mdl, norm_mode=nmode)
        print(f"{label:52s} | {r['auc_s0']:13.2f}% | {r['auc_s4']:12.2f}% | {r['coverage_pct']:9.1f}%", flush=True)

    # 2. Huang et al. Stock YOLOv8n-pose @ 640 (Default Tracker)
    stock_files = e00.index_pose_dir(Path("~/thesis-data/exp003/poses/L0_E0_c0.25_botsort").expanduser(), "ShanghaiTech")
    trks_stock = [e00.read_pose_clip(stock_files[c]) for c in ctx.names]
    r_stock = evaluate_tracks_s0_and_s4(trks_stock, ctx.labels, stgnf_model, norm_mode="N0")
    print(f"{'2.  Huang Stock YOLOv8n-pose @640 + Default BoT-SORT':52s} | {r_stock['auc_s0']:13.2f}% | {r_stock['auc_s4']:12.2f}% | {r_stock['coverage_pct']:9.1f}%", flush=True)

    # 3. Ours ML0 @ 640 (Default Tracker vs. Locked Two-Stage BoT-SORT, with STG-NF and LoRaD)
    ml0_def_files = e00.index_pose_dir(Path("~/thesis-data/exp020-ncnn/poses/L0_E0-ML0_c0.25_botsort").expanduser(), "ShanghaiTech")
    trks_ml0_def = [e00.read_pose_clip(ml0_def_files[c]) for c in ctx.names]
    r_ml0_def = evaluate_tracks_s0_and_s4(trks_ml0_def, ctx.labels, stgnf_model, norm_mode="N0")
    print(f"{'3a. Ours ML0 NCNN @640 + Default BoT-SORT + STG-NF':52s} | {r_ml0_def['auc_s0']:13.2f}% | {r_ml0_def['auc_s4']:12.2f}% | {r_ml0_def['coverage_pct']:9.1f}%", flush=True)

    trks_ml0_locked = e06.pose_tracks("ShanghaiTech", "ML0_edge", ctx, procs=2)
    for label, mdl, nmode in [
        ("3b. Ours ML0 NCNN @640 + 2-Stage BoT-SORT + STG-NF", stgnf_model, "N0"),
        ("3c. Ours ML0 NCNN @640 + 2-Stage BoT-SORT + LoRaD", lorad_n0_model, "N0"),
    ]:
        r = evaluate_tracks_s0_and_s4(trks_ml0_locked, ctx.labels, mdl, norm_mode=nmode)
        print(f"{label:52s} | {r['auc_s0']:13.2f}% | {r['auc_s4']:12.2f}% | {r['coverage_pct']:9.1f}%", flush=True)


if __name__ == "__main__":
    main()
