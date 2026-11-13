"""Refresh policies, token-bucket service gate, guarded detector cache, timebase-preserving tracker,
and strict causal score delivery for EXP-023.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from edgelab.pose.fill import DAMPING
from edgelab.pose.replay import Detections, build_tracker
from edgelab.pose.smoothing import SMOOTHERS
from edgelab.pose.stgnf_eval import SEG_LENS, _import_stgnf
from edgelab.pose.visual_propagation import (
    PersonTrackState,
    PropagatedPoseRecord,
    VisualObserverOutput,
    interpolate_b_delayed,
    measure_CheapMotionOnly,
    measure_visual_observer,
    propagate_tracks_p_cv,
    propagate_tracks_p_lk,
)

# Default train-calibrated normality sentinel (derived from normal ShanghaiTech/UBnormal training segments,
# never from test clips). Can be overridden from frozen_manifest.json.
DEFAULT_TRAIN_SENTINEL_NORMALITY = 1.65
DEFAULT_MAX_HOLD_FRAMES = 24


class GuardedDetectionCache:
    """Wraps per-frame cached detections and forbids reading frames not selected for detector inference.

    Enforces Mandatory Preflight Check #3:
    'Detector cache access: decisions cannot inspect detections on frames not selected for inference.'
    """

    def __init__(self, frames: list[dict], frame_ids: np.ndarray, c: float = 0.25):
        self._frames = frames
        self._frame_ids = np.asarray(frame_ids, dtype=int)
        self._id_to_idx = {int(fid): i for i, fid in enumerate(self._frame_ids)}
        self.c = float(c)
        self.authorized_frames: set[int] = set()
        self.accessed_frames: list[int] = []

    def __len__(self) -> int:
        return len(self._frames)

    def authorize_and_fetch(self, frame_idx: int, decision_granted: bool) -> dict[str, np.ndarray]:
        """Fetch thresholded detections for `frame_idx` ONLY if `decision_granted` is True."""
        if not decision_granted:
            raise PermissionError(
                f"Illegal detector cache access on frame {frame_idx}: detector inference was not granted."
            )
        if frame_idx not in self._id_to_idx:
            raise KeyError(f"Frame {frame_idx} not found in detection cache.")
        self.authorized_frames.add(int(frame_idx))
        self.accessed_frames.append(int(frame_idx))
        raw = self._frames[self._id_to_idx[frame_idx]]
        above = raw["conf"] >= self.c
        return {
            "boxes": raw["boxes"][above].astype(np.float64),
            "conf": raw["conf"][above].astype(np.float64),
            "cls": raw["cls"][above].astype(np.float64),
            "keypoints": raw["keypoints"][above].astype(np.float64),
        }


class TimebasePreservingTracker:
    """BoT-SORT wrapper that preserves real frame timestamps across unobserved frames.

    Enforces Mandatory Preflight Checks #4 & #5:
    - Skipped frames must not compress elapsed time.
    - Advances Kalman state (`multi_predict`) across unobserved frames while zeroing width/height velocity
      so boxes translate at their physical velocity without aspect collapse.
    - Keeps unconfirmed tracks eligible for confirmation across sparse intervals up to `max_confirm_gap`
      real frames, while expiring lost tracks strictly by real elapsed frames (`max_time_lost`).
    - Supports full state reset on scene cuts.
    """

    def __init__(
        self,
        c: float = 0.25,
        tracker_type: str = "botsort",
        max_confirm_gap: int = 8,
        max_frames_lost: int = 30,
    ):
        self.c = float(c)
        self.tracker_type = tracker_type
        self.max_confirm_gap = int(max_confirm_gap)
        self.max_frames_lost = int(max_frames_lost)
        self._reset_internal()

    def _reset_internal(self) -> None:
        self.tracker = build_tracker(self.c, self.tracker_type)
        self.tracker.max_frames_lost = self.max_frames_lost
        self.last_stepped_frame: int = -1
        self.last_observed_frame: int = -1

    def reset_on_scene_cut(self) -> None:
        """Invalidate all tracker states on a scene cut."""
        self._reset_internal()

    def step_unobserved(self, frame_idx: int) -> None:
        """Advance tracker clock and Kalman predictions for an unobserved frame `frame_idx`."""
        from ultralytics.trackers.byte_tracker import joint_stracks

        if frame_idx <= self.last_stepped_frame:
            return

        while self.last_stepped_frame < frame_idx:
            self.last_stepped_frame += 1
            self.tracker.frame_id = self.last_stepped_frame + 1

            unconfirmed, tracked = self.tracker._split_tracked()
            pool = joint_stracks(tracked, self.tracker.lost_stracks)
            # Predict active + lost tracks by 1 real frame step; freeze vw, vh during coasting
            for st in pool:
                if st.mean is not None and len(st.mean) >= 8:
                    st.mean[6] = 0.0
                    st.mean[7] = 0.0
            self.tracker.multi_predict(pool)

            # Also step unconfirmed tracks so their boxes move with initial velocity if any
            if unconfirmed:
                for st in unconfirmed:
                    if st.mean is not None and len(st.mean) >= 8:
                        st.mean[6] = 0.0
                        st.mean[7] = 0.0
                self.tracker.multi_predict(unconfirmed)

            # Expire unconfirmed tracks that exceeded max_confirm_gap in real frames
            kept_tracked = []
            for st in self.tracker.tracked_stracks:
                if not st.is_activated:
                    if (self.tracker.frame_id - st.end_frame) <= self.max_confirm_gap:
                        kept_tracked.append(st)
                    else:
                        st.mark_removed()
                else:
                    if (self.tracker.frame_id - st.end_frame) <= self.tracker.max_frames_lost:
                        kept_tracked.append(st)
                    else:
                        st.mark_removed()
            self.tracker.tracked_stracks = kept_tracked

            # Expire lost tracks that exceeded max_frames_lost in real frames
            kept_lost = []
            for st in self.tracker.lost_stracks:
                if (self.tracker.frame_id - st.end_frame) <= self.tracker.max_frames_lost:
                    kept_lost.append(st)
                else:
                    st.mark_removed()
            self.tracker.lost_stracks = kept_lost

    def observe(self, frame_idx: int, det_dict: dict[str, np.ndarray]) -> list[tuple[int, np.ndarray, float, np.ndarray]]:
        """Run tracker association on a real detector observation at `frame_idx`.

        First advances any intermediate unobserved frames `< frame_idx` so elapsed time is never compressed.
        Returns list of `(track_id, keypoints (17,3), score, det_box (4,))`.
        """
        if frame_idx > self.last_stepped_frame + 1:
            self.step_unobserved(frame_idx - 1)

        is_first_observation = self.last_observed_frame < 0
        # Sync tracker.frame_id so tracker.update() increments it to `frame_idx + 1`
        self.tracker.frame_id = frame_idx

        dets = Detections(det_dict["boxes"], det_dict["conf"], det_dict["cls"])
        tracks = self.tracker.update(dets, None)

        # If this is the very first detector observation of the clip (even at nonzero phase frame_idx > 0),
        # activate newly initiated tracks immediately just as BYTETracker does at frame_id == 1.
        if is_first_observation and frame_idx > 0 and len(tracks) == 0 and self.tracker.tracked_stracks:
            for st in self.tracker.tracked_stracks:
                st.is_activated = True
            tracks = self.tracker._format_output()

        self.last_stepped_frame = frame_idx
        self.last_observed_frame = frame_idx

        results: list[tuple[int, np.ndarray, float, np.ndarray]] = []
        for row in tracks:
            idx = int(row[7])
            track_id = int(row[4])
            kp = det_dict["keypoints"][idx].astype(np.float64)
            score = float(det_dict["conf"][idx])
            box = det_dict["boxes"][idx].astype(np.float64)
            results.append((track_id, kp, score, box))
        return results


@dataclass
class FrameDecisionRecord:
    """Per-frame scheduling, budget, and timing record (saved to decisions.jsonl)."""

    video: str
    method: str
    budget_ratio: float
    frame_idx: int
    arrival_ms: float
    requested: bool
    executed: bool
    trigger_reason: str
    reject_reason: str
    tokens_before: float
    tokens_after: float
    busy_until_ms: float
    source_observation_age_frames: int
    source_observation_age_ms: float
    disagreement_q80: float | None
    invalid_flow_fraction: float
    exterior_motion: float
    decode_cpu_ms: float
    observer_cpu_ms: float
    detector_cpu_ms: float
    tracker_cpu_ms: float
    scorer_cpu_ms: float
    queue_delay_ms: float


class DetectorBudgetGate:
    """Token-bucket rate controller + physical `busy_until_ms` service-time gate.

    - Refills `budget_ratio` tokens per arriving frame up to `burst_capacity`.
    - Grants a requested refresh iff `tokens >= 1.0` AND `decision_ready_ms >= busy_until_ms`.
    - Tracks rejected refreshes (`detector_busy` vs `token_exhausted`) and budget violations.
    """

    def __init__(
        self,
        budget_ratio: float,
        fps: float = 24.0,
        burst_capacity: float = 2.0,
        initial_tokens: float = 1.0,
    ):
        self.budget_ratio = float(budget_ratio)
        self.fps = float(fps)
        self.frame_interval_ms = 1000.0 / self.fps
        self.burst_capacity = max(1.0, float(burst_capacity))
        self.tokens = float(initial_tokens)
        self.busy_until_ms: float = 0.0
        self.last_refill_frame: int = 0
        self.total_requests: int = 0
        self.total_executed: int = 0
        self.rejected_busy: int = 0
        self.rejected_tokens: int = 0
        self.budget_violations: int = 0

    def refill(self, frame_idx: int) -> float:
        if frame_idx > self.last_refill_frame:
            elapsed = frame_idx - self.last_refill_frame
            self.tokens = min(self.burst_capacity, self.tokens + elapsed * self.budget_ratio)
            self.last_refill_frame = frame_idx
        return self.tokens

    def try_acquire(
        self,
        frame_idx: int,
        requested: bool,
        decision_ready_ms: float,
        detector_service_ms: float,
    ) -> tuple[bool, str, float, float, float]:
        """Attempt to execute detector inference at `decision_ready_ms`.

        Returns `(executed, reject_reason, tokens_before, tokens_after, completion_ms)`.
        """
        tokens_before = self.refill(frame_idx)
        if not requested:
            return False, "not_requested", tokens_before, self.tokens, decision_ready_ms

        self.total_requests += 1
        if decision_ready_ms + 1e-6 < self.busy_until_ms:
            self.rejected_busy += 1
            return False, "detector_busy", tokens_before, self.tokens, decision_ready_ms

        if self.tokens + 1e-9 < 1.0:
            self.rejected_tokens += 1
            return False, "token_exhausted", tokens_before, self.tokens, decision_ready_ms

        self.tokens -= 1.0
        completion_ms = decision_ready_ms + detector_service_ms
        self.busy_until_ms = completion_ms
        self.total_executed += 1
        return True, "none", tokens_before, self.tokens, completion_ms


@dataclass
class RefreshThresholds:
    """Calibrated thresholds for Motion-Only (S3) and Prediction-Validity (S4, S5) controllers."""

    motion_score_thresh: float = 0.14
    disagreement_q80_thresh: float = 0.045
    invalid_fraction_thresh: float = 0.45
    exterior_motion_thresh: float = 0.16
    max_gap_frames: int = 8
    discovery_multiplier: float = 1.5  # full-scene scan interval = round(discovery_multiplier * K)


class RefreshController:
    """Base interface for detector refresh scheduling policies."""

    def decide(
        self,
        frame_idx: int,
        last_det_frame: int | None,
        tracks: dict[int, PersonTrackState],
        tokens_available: float,
        observer_out: VisualObserverOutput | None = None,
        cheap_motion: tuple[float, float, bool] | None = None,
    ) -> tuple[bool, str]:
        raise NotImplementedError


class FixedCadenceController(RefreshController):
    """Controller 1: Fixed every K-th frame `(frame_idx - phase) % K == 0`."""

    def __init__(self, stride_k: int, phase: int = 0):
        self.stride_k = max(1, int(stride_k))
        self.phase = int(phase) % self.stride_k

    def decide(
        self,
        frame_idx: int,
        last_det_frame: int | None,
        tracks: dict[int, PersonTrackState],
        tokens_available: float,
        observer_out: VisualObserverOutput | None = None,
        cheap_motion: tuple[float, float, bool] | None = None,
    ) -> tuple[bool, str]:
        if (frame_idx - self.phase) % self.stride_k == 0:
            return True, f"fixed_k{self.stride_k}_p{self.phase}"
        return False, "fixed_skip"


class MotionOnlyController(RefreshController):
    """Controller 2 (Arm S3): Low-cost frame motion magnitude + bounded max interval."""

    def __init__(self, stride_k: int, thresholds: RefreshThresholds | None = None):
        self.stride_k = max(1, int(stride_k))
        self.th = thresholds or RefreshThresholds()
        self.max_interval = max(self.stride_k + 1, int(round(self.th.discovery_multiplier * self.stride_k)))

    def decide(
        self,
        frame_idx: int,
        last_det_frame: int | None,
        tracks: dict[int, PersonTrackState],
        tokens_available: float,
        observer_out: VisualObserverOutput | None = None,
        cheap_motion: tuple[float, float, bool] | None = None,
    ) -> tuple[bool, str]:
        if last_det_frame is None:
            return True, "initial_frame"
        age = frame_idx - last_det_frame
        if cheap_motion is not None:
            motion_score, _, scene_cut = cheap_motion
            if scene_cut and tokens_available >= 1.0:
                return True, "scene_cut"
            if motion_score >= self.th.motion_score_thresh and tokens_available >= 1.0:
                return True, "motion_magnitude"
        if age >= self.max_interval and tokens_available >= 1.0:
            return True, "max_interval_discovery"
        if age >= self.stride_k and tokens_available >= 1.5:
            return True, "token_surplus_cadence"
        return False, "calm_hold"


class PredictionValidityController(RefreshController):
    """Controller 3 (Arms S4 & S5): Prediction-validity + invalidity guard + exterior discovery."""

    def __init__(self, stride_k: int, thresholds: RefreshThresholds | None = None):
        self.stride_k = max(1, int(stride_k))
        self.th = thresholds or RefreshThresholds()
        self.discovery_interval = max(self.stride_k + 1, int(round(self.th.discovery_multiplier * self.stride_k)))

    def decide(
        self,
        frame_idx: int,
        last_det_frame: int | None,
        tracks: dict[int, PersonTrackState],
        tokens_available: float,
        observer_out: VisualObserverOutput | None = None,
        cheap_motion: tuple[float, float, bool] | None = None,
    ) -> tuple[bool, str]:
        if last_det_frame is None:
            return True, "initial_frame"

        age_since_det = frame_idx - last_det_frame
        max_track_age = (
            max((frame_idx - tr.last_obs_frame) for tr in tracks.values())
            if tracks
            else age_since_det
        )

        if observer_out is not None:
            if observer_out.scene_cut and tokens_available >= 1.0:
                return True, "scene_cut"

            # 1. Exterior motion outside existing person boxes -> new person entry cue
            if observer_out.exterior_motion_score >= self.th.exterior_motion_thresh and tokens_available >= 1.0:
                return True, "exterior_new_person"

            # 2. Failure is NOT calmness: if tracks exist but no reliable flow or high invalid joint fraction
            if len(tracks) > 0:
                if (
                    not observer_out.any_reliable_motion
                    or observer_out.max_invalid_fraction >= self.th.invalid_fraction_thresh
                ) and tokens_available >= 1.0:
                    return True, "excessive_flow_invalidity"

                # 3. High-quantile normalized displacement disagreement between P-CV and LK
                if (
                    observer_out.any_reliable_motion
                    and np.isfinite(observer_out.max_disagreement_q80)
                    and observer_out.max_disagreement_q80 >= self.th.disagreement_q80_thresh
                    and tokens_available >= 1.0
                ):
                    return True, "prediction_disagreement"

        # 4. Observation age limit / periodic full-scene discovery scan
        if max_track_age >= self.th.max_gap_frames and tokens_available >= 1.0:
            return True, "max_observation_age"
        if age_since_det >= self.discovery_interval and tokens_available >= 1.0:
            return True, "periodic_discovery_scan"

        # 5. If no tracks currently active, scan at base cadence K
        if len(tracks) == 0 and age_since_det >= self.stride_k and tokens_available >= 1.0:
            return True, "empty_scene_scan"

        # 6. Prevent token overflow (use banked capacity when age >= K and bucket is nearly full)
        if age_since_det >= self.stride_k and tokens_available >= 1.65:
            return True, "banked_token_refresh"

        return False, "prediction_valid"


class CausalLiveStgnf:
    """Dataset-aware online STG-NF scorer with explicit pose availability tracking.

    Supports both ShanghaiTech (`seg_len=24`) and UBnormal (`seg_len=16`).
    Ensures that for delayed comparators (e.g. `B-linear`), a window containing interpolated poses
    is never scored or emitted before all poses in the window are available (`max(available_at_frame)`).
    """

    def __init__(self, model, dataset: str = "ShanghaiTech", max_missing: int = 2):
        self.model = model
        self.dataset = dataset
        self.seg_len = SEG_LENS[dataset]
        self.max_missing = int(max_missing)
        # track_id -> dict[frame_idx, PropagatedPoseRecord]
        self.track_history: dict[int, dict[int, PropagatedPoseRecord]] = {}
        # Set of already-scored (track_id, start_frame) windows so each window is scored at most once
        self.scored_windows: set[tuple[int, int]] = set()

    @staticmethod
    def load_model_for_dataset(checkpoint: Path, dataset: str = "ShanghaiTech"):
        _import_stgnf()
        from args import init_parser, init_sub_args
        from models.STG_NF.model_pose import STG_NF
        from utils.train_utils import init_model_params

        seg_len = SEG_LENS[dataset]
        args = init_parser().parse_args(["--dataset", dataset, "--seg_len", str(seg_len), "--device", "cpu"])
        args, _ = init_sub_args(args)
        fake_dataset = {"test": [[np.zeros((3, seg_len, 18))]]}
        model = STG_NF(**init_model_params(args, fake_dataset))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=False)
        model.set_actnorm_init()
        return model.eval()

    def score_batch(self, segments: np.ndarray) -> np.ndarray:
        """segments (N, seg_len, 17, 3) -> normality (N,), higher = more normal (= -NLL)."""
        _import_stgnf()
        from dataset import keypoints17_to_coco18
        from utils.data_utils import normalize_pose

        x = normalize_pose(keypoints17_to_coco18(segments).astype(np.float32))
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)[:, :2])).float()
        with torch.no_grad():
            _, nll = self.model(x, label=torch.ones(len(x)))
        return -nll.numpy().reshape(-1)

    def ingest_poses(
        self,
        current_frame: int,
        records: list[PropagatedPoseRecord],
    ) -> list[tuple[int, int, int, float, float, float]]:
        """Ingest newly available pose records at `current_frame` and score any newly completed windows.

        Returns list of `(track_id, start_frame, available_frame, available_time_ms, normality, pred_share)`.
        """
        touched_tracks: set[int] = set()
        for rec in records:
            if rec.available_at_frame > current_frame:
                raise ValueError(
                    f"Causality violation: pose for frame {rec.frame_idx} is only available at "
                    f"frame {rec.available_at_frame} > current_frame {current_frame}."
                )
            th = self.track_history.setdefault(rec.track_id, {})
            th[rec.frame_idx] = rec
            touched_tracks.add(rec.track_id)

        batch_segments: list[np.ndarray] = []
        batch_meta: list[tuple[int, int, int, float, float]] = []

        for tid in sorted(touched_tracks):
            th = self.track_history[tid]
            if len(th) < self.seg_len:
                continue
            frames_sorted = sorted(th.keys())
            n_f = len(frames_sorted)
            # Only recently completed windows near the tail of the track can be new
            start_idx = max(0, n_f - self.seg_len - 16)
            for idx in range(start_idx, n_f - self.seg_len + 1):
                win_frames = frames_sorted[idx : idx + self.seg_len]
                start_f = win_frames[0]
                if (tid, start_f) in self.scored_windows:
                    continue
                # STG-NF continuity rule: at most `max_missing` missing slots in [start_f, start_f + seg_len)
                in_span = sum(1 for f in win_frames if start_f <= f < start_f + self.seg_len)
                if in_span < self.seg_len - self.max_missing:
                    continue
                win_recs = [th[f] for f in win_frames]
                avail_frame = max(max(r.available_at_frame for r in win_recs), win_frames[-1], current_frame)
                avail_time_ms = max(r.available_at_time_ms for r in win_recs)
                pred_share = sum(1 for r in win_recs if r.age > 0) / float(self.seg_len)
                seg_arr = np.stack([r.keypoints for r in win_recs], axis=0)
                batch_segments.append(seg_arr)
                batch_meta.append((tid, start_f, avail_frame, avail_time_ms, pred_share))
                self.scored_windows.add((tid, start_f))

        if not batch_segments:
            return []

        normalities = self.score_batch(np.stack(batch_segments, axis=0))
        out = []
        for (tid, start_f, avail_f, avail_ms, pred_share), norm_val in zip(batch_meta, normalities):
            out.append((tid, start_f, avail_f, avail_ms, float(norm_val), pred_share))
        return out


class CausalScoreDeliverer:
    """Strictly causal online score delivery policy with fixed train-calibrated sentinel and bounded hold.

    Enforces Mandatory Preflight Checks #1 (Prefix invariance) and #9 (Score-policy integrity):
    - Emits frame score `score[t]` at the end of frame `t` using ONLY segments available on or before `t`.
    - Holds the last valid frame normality for at most `max_hold_frames` frames.
    - Uses fixed `train_sentinel_normality` before the first scored window or after hold expiry (never uses
      future test frames or retrospective dataset max).
    - Applies causal half-Gaussian smoothing up to frame `t`.
    """

    def __init__(
        self,
        train_sentinel_normality: float = DEFAULT_TRAIN_SENTINEL_NORMALITY,
        max_hold_frames: int = DEFAULT_MAX_HOLD_FRAMES,
        smoother_name: str = "causal_half_gaussian",
    ):
        self.train_sentinel_normality = float(train_sentinel_normality)
        self.max_hold_frames = int(max_hold_frames)
        self.smoother = SMOOTHERS[smoother_name]
        self.raw_normalities: list[float] = []
        self.scorable_flags: list[bool] = []
        self.last_valid_normality: float | None = None
        self.last_valid_frame: int | None = None
        self.emitted_scores: list[float] = []

    def step_frame(self, frame_idx: int, available_normalities: list[float]) -> float:
        """Emit the causal anomaly score for `frame_idx` given segment normalities completing at `frame_idx`."""
        if frame_idx != len(self.raw_normalities):
            raise ValueError(f"Expected sequential frame {len(self.raw_normalities)}, got {frame_idx}")

        finite_vals = [float(v) for v in available_normalities if np.isfinite(v)]
        if finite_vals:
            norm_val = min(finite_vals)  # least normal person on this frame
            self.last_valid_normality = norm_val
            self.last_valid_frame = frame_idx
            self.scorable_flags.append(True)
        else:
            self.scorable_flags.append(False)
            if (
                self.last_valid_normality is not None
                and self.last_valid_frame is not None
                and (frame_idx - self.last_valid_frame) <= self.max_hold_frames
            ):
                norm_val = self.last_valid_normality
            else:
                norm_val = self.train_sentinel_normality

        self.raw_normalities.append(norm_val)
        # Causal smoother over bounded causal suffix (causal_half_gaussian has support <= 40 frames)
        tail = self.raw_normalities[-64:] if self.smoother is SMOOTHERS["causal_half_gaussian"] else self.raw_normalities
        prefix_anom = -np.asarray(tail, dtype=np.float64)
        smoothed_prefix = self.smoother(prefix_anom)
        score_t = float(smoothed_prefix[-1])
        self.emitted_scores.append(score_t)
        return score_t

    def get_scores_array(self) -> np.ndarray:
        return np.asarray(self.emitted_scores, dtype=np.float64)

    def get_scorable_mask(self) -> np.ndarray:
        return np.asarray(self.scorable_flags, dtype=bool)


@dataclass
class VideoSimulationResult:
    """Complete causal simulation output and diagnostics for one video under one policy and budget."""

    video: str
    method: str
    budget_ratio: float
    fps: float
    n_frames: int
    delivered_scores: np.ndarray  # (T,) strict causal live anomaly scores
    raw_normalities: np.ndarray  # (T,) raw frame normalities before causal smoothing
    scorable_mask: np.ndarray  # (T,) bool: whether a valid segment score landed on frame t
    observed_mask: np.ndarray  # (T,) bool: whether detector inference executed on frame t
    covered_any_pose_mask: np.ndarray  # (T,) bool: whether at least 1 pose (observed or propagated) existed
    propagated_pose_count: int
    observed_pose_count: int
    total_pose_count: int
    propagated_fraction: float
    measured_coverage: float
    scorable_coverage: float
    max_observation_age_frames: int
    max_observation_age_ms: float
    mean_invalid_flow_fraction: float
    actual_detector_hz: float
    decode_cpu_ms_mean: float
    observer_cpu_ms_mean: float
    detector_cpu_ms_mean: float
    tracker_cpu_ms_mean: float
    scorer_cpu_ms_mean: float
    total_cpu_ms_per_sec: float
    queue_p95_ms: float
    rejected_busy: int
    rejected_tokens: int
    budget_violations: int
    decisions: list[FrameDecisionRecord]
    pose_records_by_frame: dict[int, list[PropagatedPoseRecord]] = field(default_factory=dict)


DEFAULT_STAGE_COSTS_MS = {
    "decode_ms": 0.85,
    "observer_cheap_ms": 0.18,
    "observer_lk_ms": 0.62,
    "detector_ms": 13.29,  # ML0 NCNN FP16 @ 640 (384x640) measured VM service time
    "tracker_ms": 0.54,
    "scorer_per_seg_ms": 0.31,
}


def simulate_video_policy(
    video: str,
    method: str,
    budget_ratio: float,
    frames_det: list[dict],
    frame_ids: np.ndarray,
    stgnf_model,
    gray_loader=None,
    dataset: str = "ShanghaiTech",
    fps: float = 24.0,
    c: float = 0.25,
    phase: int = 0,
    max_gap: int = 8,
    damping: float = DAMPING,
    thresholds: RefreshThresholds | None = None,
    stage_costs_ms: dict[str, float] | None = None,
    train_sentinel_normality: float = DEFAULT_TRAIN_SENTINEL_NORMALITY,
    max_hold_frames: int = DEFAULT_MAX_HOLD_FRAMES,
    burst_capacity: float = 2.0,
) -> VideoSimulationResult:
    """Run causal frame-by-frame simulation of `method` on `video` at `budget_ratio`.

    Supported `method` IDs:
      - "R":  Full rate (budget_ratio = 1.0), real observations only
      - "S0": Fixed cadence, no fill (real timestamps)
      - "S0-skip": Fixed cadence, re-indexed skip compatibility control
      - "S1": Fixed cadence, P-CV causal propagation
      - "S2": Fixed cadence, P-LK causal visual propagation with bounded CV fallback
      - "S3": Motion-only adaptive refresh, P-CV causal propagation
      - "S4": Prediction-validity adaptive refresh (charged LK observer), P-CV propagation
      - "S5": Prediction-validity adaptive refresh, P-LK propagation with bounded CV fallback
      - "B":  Fixed cadence, B-linear interpolation with strict delayed availability
    """
    costs = {**DEFAULT_STAGE_COSTS_MS, **(stage_costs_ms or {})}
    th = thresholds or RefreshThresholds(max_gap_frames=max_gap)
    n_frames = len(frames_det)
    frame_interval_ms = 1000.0 / float(fps)

    if method == "R":
        eff_budget = 1.0
        stride_k = 1
        controller: RefreshController = FixedCadenceController(1, phase=0)
        observer_kind = "none"
        prop_mode = "none"
    elif method in ("S0", "S0-skip"):
        eff_budget = float(budget_ratio)
        stride_k = max(1, int(round(1.0 / eff_budget)))
        controller = FixedCadenceController(stride_k, phase=phase)
        observer_kind = "none"
        prop_mode = "skip_reindex" if method == "S0-skip" else "none"
    elif method == "S1":
        eff_budget = float(budget_ratio)
        stride_k = max(1, int(round(1.0 / eff_budget)))
        controller = FixedCadenceController(stride_k, phase=phase)
        observer_kind = "none"
        prop_mode = "P-CV"
    elif method == "S2":
        eff_budget = float(budget_ratio)
        stride_k = max(1, int(round(1.0 / eff_budget)))
        controller = FixedCadenceController(stride_k, phase=phase)
        observer_kind = "lk"
        prop_mode = "P-LK"
    elif method == "S3":
        eff_budget = float(budget_ratio)
        stride_k = max(1, int(round(1.0 / eff_budget)))
        controller = MotionOnlyController(stride_k, thresholds=th)
        observer_kind = "cheap"
        prop_mode = "P-CV"
    elif method == "S4":
        eff_budget = float(budget_ratio)
        stride_k = max(1, int(round(1.0 / eff_budget)))
        controller = PredictionValidityController(stride_k, thresholds=th)
        observer_kind = "lk"
        prop_mode = "P-CV"
    elif method == "S5":
        eff_budget = float(budget_ratio)
        stride_k = max(1, int(round(1.0 / eff_budget)))
        controller = PredictionValidityController(stride_k, thresholds=th)
        observer_kind = "lk"
        prop_mode = "P-LK"
    elif method == "B":
        eff_budget = float(budget_ratio)
        stride_k = max(1, int(round(1.0 / eff_budget)))
        controller = FixedCadenceController(stride_k, phase=phase)
        observer_kind = "none"
        prop_mode = "B-linear"
    else:
        raise ValueError(f"Unknown method {method!r}")

    if observer_kind in ("cheap", "lk") and gray_loader is None:
        raise ValueError(f"Method {method} requires `gray_loader` (actual grayscale video frames).")

    guarded_cache = GuardedDetectionCache(frames_det, frame_ids, c=c)
    tracker = TimebasePreservingTracker(c=c, max_confirm_gap=max(max_gap, stride_k * 2))
    # For fixed cadence at phase > 0, ensure initial token is ready at the first scheduled frame
    gate = DetectorBudgetGate(
        budget_ratio=eff_budget,
        fps=fps,
        burst_capacity=burst_capacity,
        initial_tokens=1.0,
    )
    scorer = CausalLiveStgnf(stgnf_model, dataset=dataset)
    deliverer = CausalScoreDeliverer(
        train_sentinel_normality=train_sentinel_normality,
        max_hold_frames=max_hold_frames,
    )

    active_tracks: dict[int, PersonTrackState] = {}
    b_anchors: dict[int, tuple[int, np.ndarray, float, np.ndarray]] = {}
    last_det_frame: int | None = None
    prev_gray: np.ndarray | None = None
    img_shape: tuple[int, int] | None = None

    observed_mask = np.zeros(n_frames, dtype=bool)
    covered_any_pose_mask = np.zeros(n_frames, dtype=bool)
    observed_pose_count = 0
    propagated_pose_count = 0
    max_obs_age_frames = 0
    invalid_flow_vals: list[float] = []
    decisions: list[FrameDecisionRecord] = []
    pose_records_by_frame: dict[int, list[PropagatedPoseRecord]] = {}
    processed_counter = 0

    for f in range(n_frames):
        arrival_ms = f * frame_interval_ms
        decode_ms = float(costs["decode_ms"])
        observer_ms = 0.0
        observer_out: VisualObserverOutput | None = None
        cheap_motion: tuple[float, float, bool] | None = None

        if observer_kind in ("cheap", "lk"):
            curr_gray = gray_loader(f)
            if img_shape is None:
                img_shape = curr_gray.shape[:2]
            if prev_gray is not None:
                if observer_kind == "cheap":
                    cheap_motion = measure_CheapMotionOnly(prev_gray, curr_gray)
                    observer_ms = float(costs["observer_cheap_ms"])
                    if cheap_motion[2]:  # scene_cut
                        active_tracks.clear()
                        b_anchors.clear()
                        tracker.reset_on_scene_cut()
                elif observer_kind == "lk":
                    observer_out = measure_visual_observer(
                        prev_gray, curr_gray, active_tracks, f, damping=damping
                    )
                    observer_ms = float(costs["observer_lk_ms"])
                    if len(active_tracks) > 0:
                        invalid_flow_vals.append(observer_out.mean_invalid_fraction)
                    if observer_out.scene_cut:
                        active_tracks.clear()
                        b_anchors.clear()
                        tracker.reset_on_scene_cut()
            prev_gray = curr_gray

        tokens_avail = gate.refill(f)
        requested, trigger_reason = controller.decide(
            f, last_det_frame, active_tracks, tokens_avail, observer_out, cheap_motion
        )
        decision_ready_ms = arrival_ms + decode_ms + observer_ms
        det_service_ms = float(costs["detector_ms"] + costs["tracker_ms"])

        executed, reject_reason, tok_before, tok_after, completion_ms = gate.try_acquire(
            f, requested, decision_ready_ms, det_service_ms
        )

        newly_available_records: list[PropagatedPoseRecord] = []

        if executed:
            det_ms = float(costs["detector_ms"])
            trk_ms = float(costs["tracker_ms"])
            observed_mask[f] = True
            last_det_frame = f

            det_dict = guarded_cache.authorize_and_fetch(f, decision_granted=True)
            obs_list = tracker.observe(f, det_dict)

            seen_tids: set[int] = set()
            pose_frame_idx = processed_counter if prop_mode == "skip_reindex" else f
            processed_counter += 1

            for tid, kp, score, box in obs_list:
                seen_tids.add(tid)
                if tid in active_tracks:
                    active_tracks[tid].update_from_observation(f, kp, score, box)
                else:
                    active_tracks[tid] = PersonTrackState(
                        track_id=tid,
                        last_obs_frame=f,
                        last_obs_keypoints=kp,
                        last_obs_score=score,
                        last_obs_box=box,
                    )

                # If B-linear, interpolate gap from previous anchor (now available at frame f, completion_ms!)
                if prop_mode == "B-linear" and tid in b_anchors:
                    f_prev, kp_prev, sc_prev, box_prev = b_anchors[tid]
                    interp_recs = interpolate_b_delayed(
                        track_id=tid,
                        frame_before=f_prev,
                        kp_before=kp_prev,
                        score_before=sc_prev,
                        box_before=box_prev,
                        frame_after=f,
                        kp_after=kp,
                        score_after=score,
                        box_after=box,
                        completion_time_after_ms=completion_ms,
                        max_gap=max_gap,
                    )
                    for irec in interp_recs:
                        newly_available_records.append(irec)
                        pose_records_by_frame.setdefault(irec.frame_idx, []).append(irec)
                        covered_any_pose_mask[irec.frame_idx] = True
                        propagated_pose_count += 1
                if prop_mode == "B-linear":
                    b_anchors[tid] = (f, kp.copy(), score, box.copy())

                obs_rec = PropagatedPoseRecord(
                    frame_idx=pose_frame_idx,
                    track_id=tid,
                    keypoints=kp.copy(),
                    score=score,
                    box=box.copy(),
                    age=0,
                    mode="observed",
                    valid_joint_fraction=1.0,
                    available_at_frame=pose_frame_idx if prop_mode == "skip_reindex" else f,
                    available_at_time_ms=completion_ms,
                )
                newly_available_records.append(obs_rec)
                pose_records_by_frame.setdefault(f, []).append(obs_rec)
                observed_pose_count += 1

            # For tracks that were active previously but missed detection on this observed frame:
            # Propagate if within max_gap (for P-CV / P-LK), or expire if > max_gap
            unseen_tracks = {tid: tr for tid, tr in active_tracks.items() if tid not in seen_tids}
            if unseen_tracks:
                if prop_mode == "P-CV":
                    prop_unseen = propagate_tracks_p_cv(
                        unseen_tracks, f, completion_ms, max_gap=max_gap, damping=damping, img_shape=img_shape
                    )
                    for tid in list(active_tracks.keys()):
                        if tid not in seen_tids and tid not in unseen_tracks:
                            del active_tracks[tid]
                    for rec in prop_unseen.values():
                        newly_available_records.append(rec)
                        pose_records_by_frame.setdefault(f, []).append(rec)
                        propagated_pose_count += 1
                elif prop_mode == "P-LK" and observer_out is not None:
                    prop_unseen = propagate_tracks_p_lk(
                        unseen_tracks,
                        f,
                        completion_ms,
                        observer_out,
                        max_gap=max_gap,
                        damping=damping,
                        img_shape=img_shape,
                    )
                    for tid in list(active_tracks.keys()):
                        if tid not in seen_tids and tid not in unseen_tracks:
                            del active_tracks[tid]
                    for rec in prop_unseen.values():
                        newly_available_records.append(rec)
                        pose_records_by_frame.setdefault(f, []).append(rec)
                        propagated_pose_count += 1
                else:
                    for tid in list(unseen_tracks.keys()):
                        if (f - active_tracks[tid].last_obs_frame) > max_gap:
                            del active_tracks[tid]
        else:
            det_ms = 0.0
            trk_ms = 0.0
            tracker.step_unobserved(f)

            if prop_mode == "P-CV":
                prop_map = propagate_tracks_p_cv(
                    active_tracks, f, completion_ms, max_gap=max_gap, damping=damping, img_shape=img_shape
                )
                for rec in prop_map.values():
                    newly_available_records.append(rec)
                    pose_records_by_frame.setdefault(f, []).append(rec)
                    propagated_pose_count += 1
            elif prop_mode == "P-LK" and observer_out is not None:
                prop_map = propagate_tracks_p_lk(
                    active_tracks,
                    f,
                    completion_ms,
                    observer_out,
                    max_gap=max_gap,
                    damping=damping,
                    img_shape=img_shape,
                )
                for rec in prop_map.values():
                    newly_available_records.append(rec)
                    pose_records_by_frame.setdefault(f, []).append(rec)
                    propagated_pose_count += 1
            else:
                # Expire stale tracks past max_gap
                expired_tids = [tid for tid, tr in active_tracks.items() if (f - tr.last_obs_frame) > max_gap]
                for tid in expired_tids:
                    del active_tracks[tid]

        if pose_records_by_frame.get(f):
            covered_any_pose_mask[f] = True

        # Track maximum observation age across active tracks or frames since last detector
        if active_tracks:
            curr_age_frames = max(f - tr.last_obs_frame for tr in active_tracks.values())
        elif last_det_frame is not None:
            curr_age_frames = f - last_det_frame
        else:
            curr_age_frames = 0
        max_obs_age_frames = max(max_obs_age_frames, curr_age_frames)

        # Ingest newly available poses into STG-NF scorer and emit frame score
        ingest_clock = (processed_counter - 1) if (prop_mode == "skip_reindex" and executed) else f
        if prop_mode == "skip_reindex" and not executed:
            new_segs = []
        else:
            new_segs = scorer.ingest_poses(ingest_clock, newly_available_records)
        scorer_ms = float(len(new_segs) * costs["scorer_per_seg_ms"])
        completion_ms += scorer_ms
        avail_norms = [norm_val for (_, _, _, _, norm_val, _) in new_segs]
        deliverer.step_frame(f, avail_norms)

        queue_delay_ms = max(0.0, completion_ms - arrival_ms)
        dis_val = (
            float(observer_out.max_disagreement_q80)
            if (observer_out is not None and np.isfinite(observer_out.max_disagreement_q80))
            else None
        )
        inv_val = float(observer_out.max_invalid_fraction) if observer_out is not None else 0.0
        ext_val = (
            float(observer_out.exterior_motion_score)
            if observer_out is not None
            else (float(cheap_motion[0]) if cheap_motion is not None else 0.0)
        )

        decisions.append(
            FrameDecisionRecord(
                video=video,
                method=method,
                budget_ratio=eff_budget,
                frame_idx=f,
                arrival_ms=arrival_ms,
                requested=requested,
                executed=executed,
                trigger_reason=trigger_reason,
                reject_reason=reject_reason,
                tokens_before=tok_before,
                tokens_after=tok_after,
                busy_until_ms=gate.busy_until_ms,
                source_observation_age_frames=curr_age_frames,
                source_observation_age_ms=curr_age_frames * frame_interval_ms,
                disagreement_q80=dis_val,
                invalid_flow_fraction=inv_val,
                exterior_motion=ext_val,
                decode_cpu_ms=decode_ms,
                observer_cpu_ms=observer_ms,
                detector_cpu_ms=det_ms,
                tracker_cpu_ms=trk_ms,
                scorer_cpu_ms=scorer_ms,
                queue_delay_ms=queue_delay_ms,
            )
        )

    total_poses = observed_pose_count + propagated_pose_count
    prop_frac = float(propagated_pose_count / total_poses) if total_poses > 0 else 0.0
    scorable_mask = deliverer.get_scorable_mask()
    measured_cov = float(np.mean(observed_mask)) if n_frames > 0 else 0.0
    scorable_cov = float(np.mean(scorable_mask)) if n_frames > 0 else 0.0
    actual_det_hz = float(np.mean(observed_mask) * fps) if n_frames > 0 else 0.0

    dec_arr = np.array([d.decode_cpu_ms for d in decisions], dtype=np.float64)
    obs_arr = np.array([d.observer_cpu_ms for d in decisions], dtype=np.float64)
    det_arr = np.array([d.detector_cpu_ms for d in decisions], dtype=np.float64)
    trk_arr = np.array([d.tracker_cpu_ms for d in decisions], dtype=np.float64)
    scr_arr = np.array([d.scorer_cpu_ms for d in decisions], dtype=np.float64)
    q_arr = np.array([d.queue_delay_ms for d in decisions], dtype=np.float64)

    total_cpu_per_frame = float(np.mean(dec_arr + obs_arr + det_arr + trk_arr + scr_arr)) if n_frames > 0 else 0.0
    total_cpu_ms_per_sec = total_cpu_per_frame * float(fps)

    # Check if actual detector calls exceeded budget + burst allowance
    max_allowed_calls = int(np.ceil(n_frames * eff_budget + gate.burst_capacity))
    if int(np.sum(observed_mask)) > max_allowed_calls:
        gate.budget_violations += int(np.sum(observed_mask)) - max_allowed_calls

    return VideoSimulationResult(
        video=video,
        method=method,
        budget_ratio=eff_budget,
        fps=fps,
        n_frames=n_frames,
        delivered_scores=deliverer.get_scores_array(),
        raw_normalities=np.asarray(deliverer.raw_normalities, dtype=np.float64),
        scorable_mask=scorable_mask,
        observed_mask=observed_mask,
        covered_any_pose_mask=covered_any_pose_mask,
        propagated_pose_count=propagated_pose_count,
        observed_pose_count=observed_pose_count,
        total_pose_count=total_poses,
        propagated_fraction=prop_frac,
        measured_coverage=measured_cov,
        scorable_coverage=scorable_cov,
        max_observation_age_frames=max_obs_age_frames,
        max_observation_age_ms=max_obs_age_frames * frame_interval_ms,
        mean_invalid_flow_fraction=float(np.mean(invalid_flow_vals)) if invalid_flow_vals else 0.0,
        actual_detector_hz=actual_det_hz,
        decode_cpu_ms_mean=float(np.mean(dec_arr)) if n_frames > 0 else 0.0,
        observer_cpu_ms_mean=float(np.mean(obs_arr)) if n_frames > 0 else 0.0,
        detector_cpu_ms_mean=float(np.mean(det_arr)) if n_frames > 0 else 0.0,
        tracker_cpu_ms_mean=float(np.mean(trk_arr)) if n_frames > 0 else 0.0,
        scorer_cpu_ms_mean=float(np.mean(scr_arr)) if n_frames > 0 else 0.0,
        total_cpu_ms_per_sec=total_cpu_ms_per_sec,
        queue_p95_ms=float(np.quantile(q_arr, 0.95)) if n_frames > 0 else 0.0,
        rejected_busy=gate.rejected_busy,
        rejected_tokens=gate.rejected_tokens,
        budget_violations=gate.budget_violations,
        decisions=decisions,
        pose_records_by_frame=pose_records_by_frame,
    )

