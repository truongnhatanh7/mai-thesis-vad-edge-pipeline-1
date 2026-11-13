"""Causal visual pose propagation and delayed B-interpolation comparator for EXP-023.

Implements:
- P-CV: causal damped constant-velocity extrapolation from the last two real detector observations.
- P-LK: pyramidal sparse Lucas-Kanade visual propagation with forward-backward consistency,
  global motion compensation, scene-cut invalidation, per-joint validity flags, and bounded CV fallback.
- B-linear (delayed): linear interpolation between two observed detector anchors, with explicit
  availability timestamps so no interpolated pose or score is ever available before the second anchor
  completes detector inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from edgelab.pose.fill import DAMPING


@dataclass
class PersonTrackState:
    """Causal state of one tracked person between expensive detector observations."""

    track_id: int
    last_obs_frame: int
    last_obs_keypoints: np.ndarray  # (17, 3) [x, y, conf]
    last_obs_score: float
    last_obs_box: np.ndarray  # (4,) [x1, y1, x2, y2]
    prev_obs_frame: int | None = None
    prev_obs_keypoints: np.ndarray | None = None
    curr_frame: int = 0
    curr_keypoints: np.ndarray = field(init=False)
    curr_score: float = field(init=False)
    curr_box: np.ndarray = field(init=False)
    joint_valid: np.ndarray = field(init=False)  # (17,) bool

    def __post_init__(self):
        self.last_obs_keypoints = np.asarray(self.last_obs_keypoints, dtype=np.float64).reshape(17, 3).copy()
        self.last_obs_box = np.asarray(self.last_obs_box, dtype=np.float64).reshape(4).copy()
        if self.prev_obs_keypoints is not None:
            self.prev_obs_keypoints = np.asarray(self.prev_obs_keypoints, dtype=np.float64).reshape(17, 3).copy()
        self.curr_frame = int(self.last_obs_frame)
        self.curr_keypoints = self.last_obs_keypoints.copy()
        self.curr_score = float(self.last_obs_score)
        self.curr_box = self.last_obs_box.copy()
        self.joint_valid = self.last_obs_keypoints[:, 2] > 0.0

    def update_from_observation(
        self,
        frame_idx: int,
        keypoints: np.ndarray,
        score: float,
        box: np.ndarray,
    ) -> None:
        """Update track state when a fresh detector observation is associated to this track."""
        if frame_idx > self.last_obs_frame:
            self.prev_obs_frame = self.last_obs_frame
            self.prev_obs_keypoints = self.last_obs_keypoints.copy()
        self.last_obs_frame = int(frame_idx)
        self.last_obs_keypoints = np.asarray(keypoints, dtype=np.float64).reshape(17, 3).copy()
        self.last_obs_score = float(score)
        self.last_obs_box = np.asarray(box, dtype=np.float64).reshape(4).copy()
        self.curr_frame = int(frame_idx)
        self.curr_keypoints = self.last_obs_keypoints.copy()
        self.curr_score = float(score)
        self.curr_box = self.last_obs_box.copy()
        self.joint_valid = self.last_obs_keypoints[:, 2] > 0.0

    def velocity_cv(self) -> np.ndarray:
        """Per-frame (17, 2) constant-velocity vector from the last two real detector observations."""
        if (
            self.prev_obs_frame is None
            or self.prev_obs_keypoints is None
            or self.last_obs_frame <= self.prev_obs_frame
        ):
            return np.zeros((17, 2), dtype=np.float64)
        dt = float(self.last_obs_frame - self.prev_obs_frame)
        return (self.last_obs_keypoints[:, :2] - self.prev_obs_keypoints[:, :2]) / dt

    def person_height(self) -> float:
        """Observed height of the person in pixels (at least 16.0 px to prevent division blow-up)."""
        box_h = float(self.last_obs_box[3] - self.last_obs_box[1])
        vis = self.last_obs_keypoints[:, 2] > 0.05
        kp_h = float(np.ptp(self.last_obs_keypoints[vis, 1])) if np.any(vis) else 0.0
        return max(box_h, kp_h, 16.0)


@dataclass
class PersonFlowMeasurement:
    """LK optical flow measurement for one tracked person on the current frame."""

    track_id: int
    joint_dx_dy: np.ndarray  # (17, 2) measured visual displacement in image coords
    joint_valid: np.ndarray  # (17,) bool: whether forward-backward LK succeeded on this joint
    active_joints: int
    valid_joints: int
    invalid_fraction: float
    disagreement_q80: float  # NaN when valid_joints == 0 (failure is NOT zero disagreement)
    has_reliable_motion: bool


@dataclass
class VisualObserverOutput:
    """Aggregated causal visual measurement from prev_gray -> curr_gray."""

    frame_idx: int
    per_person: dict[int, PersonFlowMeasurement]
    global_shift_xy: tuple[float, float]
    global_luma_diff: float
    scene_cut: bool
    exterior_motion_score: float
    frame_motion_score: float
    max_disagreement_q80: float  # NaN if no person has reliable motion
    mean_invalid_fraction: float
    max_invalid_fraction: float
    any_reliable_motion: bool


@dataclass
class PropagatedPoseRecord:
    """A single person pose at a frame, with provenance and causal availability metadata."""

    frame_idx: int
    track_id: int
    keypoints: np.ndarray  # (17, 3)
    score: float
    box: np.ndarray  # (4,)
    age: int  # 0 if observed by detector on frame_idx, >0 if propagated/interpolated
    mode: str  # "observed", "P-CV", "P-LK", "B-linear"
    valid_joint_fraction: float
    available_at_frame: int  # earliest frame index when this pose exists
    available_at_time_ms: float  # earliest wall-clock simulation time (ms) when this pose exists


def _damped_cumulative_factor(k: int, damping: float = DAMPING) -> float:
    """Sum_{j=1..k} damping^j."""
    if k <= 0:
        return 0.0
    if abs(damping - 1.0) < 1e-9:
        return float(k)
    return float(damping * (1.0 - damping**k) / (1.0 - damping))


def _box_from_keypoints_or_shift(
    prev_box: np.ndarray,
    prev_kp: np.ndarray,
    new_kp: np.ndarray,
    img_w: int | None = None,
    img_h: int | None = None,
) -> np.ndarray:
    """Shift bounding box by median keypoint displacement."""
    vis = new_kp[:, 2] > 0.01
    if np.any(vis):
        delta = np.median(new_kp[vis, :2] - prev_kp[vis, :2], axis=0)
    else:
        delta = np.zeros(2, dtype=np.float64)
    box = prev_box.copy()
    box[0] += delta[0]
    box[2] += delta[0]
    box[1] += delta[1]
    box[3] += delta[1]
    if img_w is not None and img_h is not None:
        box[0] = np.clip(box[0], 0.0, img_w - 1.0)
        box[2] = np.clip(box[2], box[0] + 1.0, float(img_w))
        box[1] = np.clip(box[1], 0.0, img_h - 1.0)
        box[3] = np.clip(box[3], box[1] + 1.0, float(img_h))
    return box


def measure_CheapMotionOnly(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    downsample: int = 4,
) -> tuple[float, float, bool]:
    """Low-cost frame motion statistic for S3 (Motion-only controller).

    Compensates global median luma change so uniform lighting flicker is separated from motion.
    Returns (motion_score, global_luma_diff, scene_cut).
    """
    p_small = prev_gray[::downsample, ::downsample].astype(np.float32)
    c_small = curr_gray[::downsample, ::downsample].astype(np.float32)
    raw_diff = c_small - p_small
    luma_shift = float(np.median(raw_diff))
    global_luma_diff = abs(luma_shift)
    comp_abs_diff = np.abs(raw_diff - luma_shift)
    mean_raw_abs = float(np.mean(np.abs(raw_diff)))
    scene_cut = bool(global_luma_diff > 45.0 or mean_raw_abs > 55.0)
    # High-quantile (90th percentile) + fraction of pixels moving > 12 gray levels
    q90 = float(np.quantile(comp_abs_diff, 0.90))
    active_frac = float(np.mean(comp_abs_diff > 12.0))
    motion_score = (q90 / 255.0) + 2.0 * active_frac
    return motion_score, global_luma_diff, scene_cut


def measure_visual_observer(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    tracks: dict[int, PersonTrackState],
    frame_idx: int,
    damping: float = DAMPING,
    fb_thresh: float = 1.5,
    win_size: tuple[int, int] = (15, 15),
    max_level: int = 2,
    min_eig_thresh: float = 1e-3,
    min_kp_conf: float = 0.05,
    downsample: int = 4,
) -> VisualObserverOutput:
    """Run causal pyramidal LK + exterior change observer on (prev_gray, curr_gray).

    1. Downsampled global luma diff & exterior change check outside dilated person boxes.
    2. Background grid + per-person 5-point joint stencils batched into a single forward/backward
       pyramidal LK pass.
    3. Forward-backward consistency check (old -> current -> old <= fb_thresh) + local gradient
       check so textureless patches are flagged invalid rather than zero-motion.
    4. Normalized displacement disagreement (80th percentile over reliable joints) between
       P-CV predicted step and camera-motion-compensated LK displacement.
    """
    h_img, w_img = curr_gray.shape[:2]

    # Step 1: Downsampled exterior & global change analysis
    p_small = prev_gray[::downsample, ::downsample].astype(np.float32)
    c_small = curr_gray[::downsample, ::downsample].astype(np.float32)
    hs, ws = c_small.shape[:2]
    raw_diff = c_small - p_small
    luma_shift = float(np.median(raw_diff))
    global_luma_diff = abs(luma_shift)
    comp_abs_diff = np.abs(raw_diff - luma_shift)

    exterior_mask = np.ones((hs, ws), dtype=bool)
    for tr in tracks.values():
        x1, y1, x2, y2 = tr.curr_box
        bw, bh = max(x2 - x1, 8.0), max(y2 - y1, 16.0)
        pad_x, pad_y = 0.15 * bw, 0.15 * bh
        ix1 = max(0, int((x1 - pad_x) / downsample))
        iy1 = max(0, int((y1 - pad_y) / downsample))
        ix2 = min(ws, int(np.ceil((x2 + pad_x) / downsample)))
        iy2 = min(hs, int(np.ceil((y2 + pad_y) / downsample)))
        exterior_mask[iy1:iy2, ix1:ix2] = False

    if np.any(exterior_mask):
        ext_vals = comp_abs_diff[exterior_mask]
        ext_q95 = float(np.quantile(ext_vals, 0.95))
        ext_frac = float(np.mean(ext_vals > 15.0))
        exterior_motion_score = (ext_q95 / 255.0) + 2.5 * ext_frac
    else:
        exterior_motion_score = 0.0

    frame_q90 = float(np.quantile(comp_abs_diff, 0.90))
    frame_frac = float(np.mean(comp_abs_diff > 12.0))
    frame_motion_score = (frame_q90 / 255.0) + 2.0 * frame_frac

    # Step 2: Build batched points for background grid + person joint stencils
    # Background grid (4x4 = 16 points outside boxes) for global camera translation
    gx_coords = np.linspace(0.15 * w_img, 0.85 * w_img, 4)
    gy_coords = np.linspace(0.15 * h_img, 0.85 * h_img, 4)
    bg_pts = []
    for gy in gy_coords:
        for gx in gx_coords:
            inside_any = False
            for tr in tracks.values():
                if tr.curr_box[0] <= gx <= tr.curr_box[2] and tr.curr_box[1] <= gy <= tr.curr_box[3]:
                    inside_any = True
                    break
            if not inside_any:
                bg_pts.append([gx, gy])
    n_bg = len(bg_pts)

    # Person stencil points: 5 points per joint (center + 4 cardinal offsets)
    person_ids = sorted(tracks.keys())
    stencil_pts = list(bg_pts)
    person_slices: dict[int, tuple[int, np.ndarray]] = {}
    for tid in person_ids:
        tr = tracks[tid]
        h_p = tr.person_height()
        delta = float(np.clip(0.03 * h_p, 2.0, 8.0))
        offsets = np.array(
            [[0.0, 0.0], [-delta, 0.0], [delta, 0.0], [0.0, -delta], [0.0, delta]],
            dtype=np.float32,
        )
        kp_xy = tr.curr_keypoints[:, :2].astype(np.float32)  # (17, 2)
        # (17, 5, 2) -> (85, 2)
        pts_person = kp_xy[:, None, :] + offsets[None, :, :]
        pts_person[:, :, 0] = np.clip(pts_person[:, :, 0], 1.0, w_img - 2.0)
        pts_person[:, :, 1] = np.clip(pts_person[:, :, 1], 1.0, h_img - 2.0)
        start_idx = len(stencil_pts)
        stencil_pts.extend(pts_person.reshape(-1, 2).tolist())
        active_mask = tr.curr_keypoints[:, 2] >= min_kp_conf
        person_slices[tid] = (start_idx, active_mask)

    if len(stencil_pts) == 0:
        scene_cut = bool(global_luma_diff > 45.0 or float(np.mean(np.abs(raw_diff))) > 55.0)
        return VisualObserverOutput(
            frame_idx=frame_idx,
            per_person={},
            global_shift_xy=(0.0, 0.0),
            global_luma_diff=global_luma_diff,
            scene_cut=scene_cut,
            exterior_motion_score=exterior_motion_score,
            frame_motion_score=frame_motion_score,
            max_disagreement_q80=np.nan,
            mean_invalid_fraction=0.0,
            max_invalid_fraction=0.0,
            any_reliable_motion=False,
        )

    pts_prev = np.asarray(stencil_pts, dtype=np.float32).reshape(-1, 1, 2)
    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        minEigThreshold=min_eig_thresh,
    )
    pts_curr, status_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None, **lk_params)
    pts_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, pts_curr, None, **lk_params)

    pts_prev_2d = pts_prev.reshape(-1, 2)
    pts_curr_2d = pts_curr.reshape(-1, 2) if pts_curr is not None else np.full_like(pts_prev_2d, np.nan)
    pts_back_2d = pts_back.reshape(-1, 2) if pts_back is not None else np.full_like(pts_prev_2d, np.nan)
    st_fwd = status_fwd.reshape(-1).astype(bool) if status_fwd is not None else np.zeros(len(pts_prev_2d), dtype=bool)
    st_bwd = status_bwd.reshape(-1).astype(bool) if status_bwd is not None else np.zeros(len(pts_prev_2d), dtype=bool)

    fb_err = np.linalg.norm(pts_prev_2d - pts_back_2d, axis=1)
    in_bounds = (
        np.isfinite(pts_curr_2d[:, 0])
        & np.isfinite(pts_curr_2d[:, 1])
        & (pts_curr_2d[:, 0] >= 0.0)
        & (pts_curr_2d[:, 0] < w_img)
        & (pts_curr_2d[:, 1] >= 0.0)
        & (pts_curr_2d[:, 1] < h_img)
    )

    # Local spatial texture check on prev_gray so flat/uniform patches never pass as valid zero-flow
    ix = np.clip( np.round(pts_prev_2d[:, 0]).astype(int), 1, w_img - 2 )
    iy = np.clip( np.round(pts_prev_2d[:, 1]).astype(int), 1, h_img - 2 )
    gx_local = np.abs(prev_gray[iy, ix + 1].astype(np.int16) - prev_gray[iy, ix - 1].astype(np.int16))
    gy_local = np.abs(prev_gray[iy + 1, ix].astype(np.int16) - prev_gray[iy - 1, ix].astype(np.int16))
    # Also check 3x3 patch range
    patch_textured = (gx_local + gy_local) >= 2

    valid_pt = st_fwd & st_bwd & (fb_err <= fb_thresh) & in_bounds & patch_textured
    disp_all = pts_curr_2d - pts_prev_2d

    # Background global shift
    if n_bg > 0 and np.any(valid_pt[:n_bg]):
        bg_disp = disp_all[:n_bg][valid_pt[:n_bg]]
        global_dx = float(np.median(bg_disp[:, 0]))
        global_dy = float(np.median(bg_disp[:, 1]))
        bg_fail_rate = 1.0 - float(np.mean(valid_pt[:n_bg]))
    else:
        global_dx, global_dy = 0.0, 0.0
        bg_fail_rate = 1.0 if n_bg > 0 else 0.0

    mean_raw_abs = float(np.mean(np.abs(raw_diff)))
    scene_cut = bool(
        global_luma_diff > 45.0
        or mean_raw_abs > 55.0
        or (mean_raw_abs > 30.0 and bg_fail_rate > 0.85)
    )

    # Step 3: Per-person joint measurements & disagreement with P-CV
    per_person: dict[int, PersonFlowMeasurement] = {}
    disagreements = []
    invalid_fractions = []
    any_reliable = False

    for tid in person_ids:
        tr = tracks[tid]
        start_idx, active_mask = person_slices[tid]
        pts_disp = disp_all[start_idx : start_idx + 85].reshape(17, 5, 2)
        pts_val = valid_pt[start_idx : start_idx + 85].reshape(17, 5)

        joint_dx_dy = np.zeros((17, 2), dtype=np.float64)
        joint_valid = np.zeros(17, dtype=bool)

        for j in range(17):
            if not active_mask[j]:
                continue
            v_mask = pts_val[j]
            if np.any(v_mask):
                joint_valid[j] = True
                joint_dx_dy[j] = np.median(pts_disp[j, v_mask], axis=0)

        active_count = int(np.sum(active_mask))
        valid_count = int(np.sum(joint_valid & active_mask))
        if active_count > 0:
            invalid_frac = 1.0 - (valid_count / active_count)
        else:
            invalid_frac = 1.0

        # Predicted single-step P-CV displacement for frame_idx
        k_step = max(1, int(frame_idx - tr.last_obs_frame))
        dt_step = max(1, int(frame_idx - tr.curr_frame))
        vel_cv = tr.velocity_cv()
        pred_step_cv = vel_cv * (damping**k_step) * dt_step

        h_p = tr.person_height()
        reliable_mask = joint_valid & active_mask
        if np.any(reliable_mask):
            # Compensate global camera shift when comparing against person's P-CV motion
            comp_lk = joint_dx_dy[reliable_mask] - np.array([global_dx, global_dy])
            diff_vec = comp_lk - pred_step_cv[reliable_mask]
            norm_err = np.linalg.norm(diff_vec, axis=1) / h_p
            disagreement_q80 = float(np.quantile(norm_err, 0.80))
            has_reliable = True
            any_reliable = True
            disagreements.append(disagreement_q80)
        else:
            # Crucial preflight invariant: failure is NOT calmness (never 0.0 when no flow points succeeded)
            disagreement_q80 = float("nan")
            has_reliable = False

        invalid_fractions.append(invalid_frac)
        per_person[tid] = PersonFlowMeasurement(
            track_id=tid,
            joint_dx_dy=joint_dx_dy,
            joint_valid=joint_valid,
            active_joints=active_count,
            valid_joints=valid_count,
            invalid_fraction=invalid_frac,
            disagreement_q80=disagreement_q80,
            has_reliable_motion=has_reliable,
        )

    max_disagreement = float(np.max(disagreements)) if disagreements else (float("nan") if person_ids else 0.0)
    mean_invalid = float(np.mean(invalid_fractions)) if invalid_fractions else 0.0
    max_invalid = float(np.max(invalid_fractions)) if invalid_fractions else 0.0

    return VisualObserverOutput(
        frame_idx=frame_idx,
        per_person=per_person,
        global_shift_xy=(global_dx, global_dy),
        global_luma_diff=global_luma_diff,
        scene_cut=scene_cut,
        exterior_motion_score=exterior_motion_score,
        frame_motion_score=frame_motion_score,
        max_disagreement_q80=max_disagreement,
        mean_invalid_fraction=mean_invalid,
        max_invalid_fraction=max_invalid,
        any_reliable_motion=any_reliable,
    )


def propagate_tracks_p_cv(
    tracks: dict[int, PersonTrackState],
    frame_idx: int,
    current_time_ms: float,
    max_gap: int = 8,
    damping: float = DAMPING,
    img_shape: tuple[int, int] | None = None,
) -> dict[int, PropagatedPoseRecord]:
    """Causal damped constant-velocity propagation (P-CV) up to `max_gap` frames past last observation.

    Tracks older than `max_gap` are expired and removed so stale poses are never extrapolated forever.
    """
    expired = [tid for tid, tr in tracks.items() if (frame_idx - tr.last_obs_frame) > max_gap]
    for tid in expired:
        del tracks[tid]

    img_h, img_w = img_shape[:2] if img_shape is not None else (None, None)
    out: dict[int, PropagatedPoseRecord] = {}
    for tid, tr in list(tracks.items()):
        k = int(frame_idx - tr.last_obs_frame)
        if k <= 0 or k > max_gap:
            continue
        vel = tr.velocity_cv()
        factor = _damped_cumulative_factor(k, damping)
        new_xy = tr.last_obs_keypoints[:, :2] + vel * factor
        if img_w is not None and img_h is not None:
            new_xy[:, 0] = np.clip(new_xy[:, 0], 0.0, img_w - 1.0)
            new_xy[:, 1] = np.clip(new_xy[:, 1], 0.0, img_h - 1.0)
        decay = float(damping**k)
        new_conf = tr.last_obs_keypoints[:, 2] * decay
        new_kp = np.column_stack([new_xy, new_conf])
        new_score = float(tr.last_obs_score * decay)
        new_box = _box_from_keypoints_or_shift(tr.curr_box, tr.curr_keypoints, new_kp, img_w, img_h)

        tr.curr_frame = frame_idx
        tr.curr_keypoints = new_kp
        tr.curr_score = new_score
        tr.curr_box = new_box
        tr.joint_valid = tr.last_obs_keypoints[:, 2] > 0.0

        out[tid] = PropagatedPoseRecord(
            frame_idx=frame_idx,
            track_id=tid,
            keypoints=new_kp.copy(),
            score=new_score,
            box=new_box.copy(),
            age=k,
            mode="P-CV",
            valid_joint_fraction=1.0,
            available_at_frame=frame_idx,
            available_at_time_ms=current_time_ms,
        )
    return out


def propagate_tracks_p_lk(
    tracks: dict[int, PersonTrackState],
    frame_idx: int,
    current_time_ms: float,
    observer_out: VisualObserverOutput,
    max_gap: int = 8,
    damping: float = DAMPING,
    max_lk_step_ratio: float = 0.35,
    max_cum_dev_ratio: float = 0.60,
    img_shape: tuple[int, int] | None = None,
) -> dict[int, PropagatedPoseRecord]:
    """Causal visual propagation (P-LK) with forward-backward consistency and bounded P-CV fallback.

    - Scene cuts immediately invalidate all old track states.
    - Tracks with age > max_gap are expired (no eternal resurrection).
    - Reliable joints move by measured LK displacement (bounded per step and bounded in cumulative
      deviation from P-CV).
    - Unreliable joints fall back to the damped P-CV step and record joint_valid[j] = False.
    """
    if observer_out.scene_cut:
        tracks.clear()
        return {}

    expired = [tid for tid, tr in tracks.items() if (frame_idx - tr.last_obs_frame) > max_gap]
    for tid in expired:
        del tracks[tid]

    img_h, img_w = img_shape[:2] if img_shape is not None else (None, None)
    out: dict[int, PropagatedPoseRecord] = {}

    for tid, tr in list(tracks.items()):
        k = int(frame_idx - tr.last_obs_frame)
        if k <= 0 or k > max_gap:
            continue

        vel = tr.velocity_cv()
        h_p = tr.person_height()
        cv_step = vel * (damping**k)
        cv_cum_xy = tr.last_obs_keypoints[:, :2] + vel * _damped_cumulative_factor(k, damping)

        meas = observer_out.per_person.get(tid)
        prev_xy = tr.curr_keypoints[:, :2].copy()
        new_xy = prev_xy.copy()
        j_valid = np.zeros(17, dtype=bool)

        max_step_px = max_lk_step_ratio * h_p
        max_dev_px = max_cum_dev_ratio * h_p

        for j in range(17):
            if meas is not None and meas.joint_valid[j]:
                step = meas.joint_dx_dy[j].copy()
                step_norm = float(np.linalg.norm(step))
                if step_norm > max_step_px and step_norm > 1e-6:
                    step = step * (max_step_px / step_norm)
                cand_xy = prev_xy[j] + step
                # Bound cumulative deviation relative to P-CV reference trajectory
                dev = cand_xy - cv_cum_xy[j]
                dev_norm = float(np.linalg.norm(dev))
                if dev_norm > max_dev_px and dev_norm > 1e-6:
                    cand_xy = cv_cum_xy[j] + dev * (max_dev_px / dev_norm)
                new_xy[j] = cand_xy
                j_valid[j] = True
            else:
                # Bounded P-CV fallback for unreliable/untextured joint
                new_xy[j] = prev_xy[j] + cv_step[j]
                j_valid[j] = False

        if img_w is not None and img_h is not None:
            new_xy[:, 0] = np.clip(new_xy[:, 0], 0.0, img_w - 1.0)
            new_xy[:, 1] = np.clip(new_xy[:, 1], 0.0, img_h - 1.0)

        decay = float(damping**k)
        new_conf = tr.last_obs_keypoints[:, 2] * decay
        new_kp = np.column_stack([new_xy, new_conf])
        new_score = float(tr.last_obs_score * decay)
        new_box = _box_from_keypoints_or_shift(tr.curr_box, tr.curr_keypoints, new_kp, img_w, img_h)

        tr.curr_frame = frame_idx
        tr.curr_keypoints = new_kp
        tr.curr_score = new_score
        tr.curr_box = new_box
        tr.joint_valid = j_valid

        valid_frac = (
            1.0 - meas.invalid_fraction
            if meas is not None
            else 0.0
        )
        out[tid] = PropagatedPoseRecord(
            frame_idx=frame_idx,
            track_id=tid,
            keypoints=new_kp.copy(),
            score=new_score,
            box=new_box.copy(),
            age=k,
            mode="P-LK",
            valid_joint_fraction=float(valid_frac),
            available_at_frame=frame_idx,
            available_at_time_ms=current_time_ms,
        )
    return out


def interpolate_b_delayed(
    track_id: int,
    frame_before: int,
    kp_before: np.ndarray,
    score_before: float,
    box_before: np.ndarray,
    frame_after: int,
    kp_after: np.ndarray,
    score_after: float,
    box_after: np.ndarray,
    completion_time_after_ms: float,
    max_gap: int = 8,
) -> list[PropagatedPoseRecord]:
    """Delayed look-ahead linear interpolation (B-linear) between two observed detector anchors.

    CRITICAL CAUSALITY RULE:
    Every interpolated pose in (frame_before, frame_after) has:
      - available_at_frame = frame_after
      - available_at_time_ms = completion_time_after_ms
    So no score or alert derived from an interpolated pose can ever be emitted before the second
    anchor finishes detector inference.
    """
    gap = int(frame_after - frame_before - 1)
    if gap < 1 or gap > max_gap:
        return []
    kp_a = np.asarray(kp_before, dtype=np.float64).reshape(17, 3)
    kp_b = np.asarray(kp_after, dtype=np.float64).reshape(17, 3)
    box_a = np.asarray(box_before, dtype=np.float64).reshape(4)
    box_b = np.asarray(box_after, dtype=np.float64).reshape(4)
    span = float(frame_after - frame_before)

    records: list[PropagatedPoseRecord] = []
    for f in range(frame_before + 1, frame_after):
        t = float(f - frame_before) / span
        kp_interp = (1.0 - t) * kp_a + t * kp_b
        score_interp = float((1.0 - t) * score_before + t * score_after)
        box_interp = (1.0 - t) * box_a + t * box_b
        records.append(
            PropagatedPoseRecord(
                frame_idx=f,
                track_id=track_id,
                keypoints=kp_interp,
                score=score_interp,
                box=box_interp,
                age=f - frame_before,
                mode="B-linear",
                valid_joint_fraction=1.0,
                available_at_frame=int(frame_after),
                available_at_time_ms=float(completion_time_after_ms),
            )
        )
    return records
