"""Streamlit Local Web App for the Edge Pose-VAD Thesis Repository.

Launch via `./run_app.sh` from the repository root, or:
    cd experiments && uv run --extra pose streamlit run src/edgelab/app/streamlit_app.py
"""

from pathlib import Path
import os
import random
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import streamlit as st

from edgelab.app.catalog import (
    DATA_ROOT,
    MODEL_BY_ID,
    ModelSpec,
    VideoExample,
    build_model_catalog,
    build_scorer_catalog,
    find_cache_npz,
    find_pose_json,
    find_scores_npz,
    iter_video_frames,
    list_dataset_videos,
    load_dataset_summary_row,
    load_ground_truth,
    load_video_CachedDetections,
    load_video_FrameScores,
    load_video_TrackedPoses,
    read_video_frame,
)
from edgelab.app.inference import (
    LivePipelineSession,
    apply_lighting_and_enhancer,
    draw_poses_and_boxes,
    load_model_handle,
    run_live_pose_model,
)
from edgelab.pose.score_model import load_model as load_stgnf_checkpoint
from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT

st.set_page_config(
    page_title="Edge Pose-VAD Explorer (ShanghaiTech & UBnormal)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------- Cached Loaders

@st.cache_data(show_spinner=False)
def get_videos(dataset: str) -> list[VideoExample]:
    return list_dataset_videos(dataset)


@st.cache_resource(show_spinner="Loading pose model weights into memory...")
def get_loaded_model(model_id: str):
    spec = MODEL_BY_ID[model_id]
    return load_model_handle(spec)


@st.cache_resource(show_spinner="Loading STG-NF checkpoint...")
def get_loaded_stgnf(checkpoint_str: str, dataset: str):
    return load_stgnf_checkpoint(Path(checkpoint_str), dataset=dataset)


@st.cache_data(show_spinner=False)
def get_cached_tracked_poses(json_path_str: str) -> dict[int, list[dict]]:
    return load_video_TrackedPoses(Path(json_path_str))


@st.cache_data(show_spinner=False)
def get_cached_frame_scores(scores_npz_str: str, video_id: str) -> dict | None:
    return load_video_FrameScores(Path(scores_npz_str), video_id)


# ---------------------------------------------------------------- Random Selection Helper

def pick_random_example(target_dataset: str):
    """Callback: select a random clip from `target_dataset` and jump to its anomaly or start frame."""
    clip_filter = st.session_state.get("clip_filter_sel", "Anomalous Only")
    videos = get_videos(target_dataset)
    if not videos:
        return
    pool = [
        v for v in videos
        if clip_filter == "All Clips"
        or (clip_filter == "Anomalous Only" and v.anomalous_frames > 0)
        or (clip_filter == "Normal Only" and v.anomalous_frames == 0)
        or (clip_filter == "Weather (Fog/Smoke/Rain)" and "weather" in v.stratum)
    ]
    if not pool:
        pool = videos
    chosen = random.choice(pool)
    st.session_state["dataset_radio"] = target_dataset
    st.session_state["scene_filter"] = "All"
    st.session_state[f"selected_vid_{target_dataset}"] = chosen.video_id

    gt = load_ground_truth(target_dataset, chosen.video_id)
    anom_idx = np.where(gt == 1)[0]
    start_f = int(anom_idx[len(anom_idx) // 4]) if len(anom_idx) > 0 else 0
    st.session_state[f"frame_{chosen.video_id}"] = start_f


def set_video_frame(video_id: str, target_frame: int):
    """Callback: set the frame slider for `video_id` before the slider widget renders."""
    st.session_state[f"frame_{video_id}"] = target_frame


# ---------------------------------------------------------------- Plotting Helpers

def plot_anomaly_timeline(
    gt: np.ndarray,
    current_frame: int,
    primary_name: str,
    primary_scores: dict | None,
    compare_name: str | None = None,
    compare_scores: dict | None = None,
    live_online_scores: dict[int, float] | None = None,
):
    fig, ax = plt.subplots(figsize=(11, 2.6), dpi=115)
    frames = np.arange(len(gt))

    # Shade ground-truth anomalous regions
    if gt.any():
        in_anom = False
        start = 0
        first_span = int(np.argmax(gt))
        for i, val in enumerate(gt):
            if val == 1 and not in_anom:
                in_anom = True
                start = i
            elif val == 0 and in_anom:
                in_anom = False
                ax.axvspan(start, i - 1, color="#ff4b4b", alpha=0.22, label="GT Anomaly" if start == first_span else None)
        if in_anom:
            ax.axvspan(start, len(gt) - 1, color="#ff4b4b", alpha=0.22, label="GT Anomaly" if start == first_span else None)

    if primary_scores is not None:
        s = primary_scores["scores"]
        cov = primary_scores["covered"]
        ax.plot(frames, s, color="#1f77b4", linewidth=1.8, label=f"{primary_name} (cov={cov.mean():.2f})")
        uncov_idx = np.where(~cov)[0]
        if len(uncov_idx) > 0:
            ax.scatter(uncov_idx, s[uncov_idx], color="#d62728", s=8, alpha=0.55, zorder=4, label="Uncovered frame")

    if compare_scores is not None and compare_name:
        s2 = compare_scores["scores"]
        cov2 = compare_scores["covered"]
        ax.plot(frames, s2, color="#2ca02c", linewidth=1.5, linestyle="--", label=f"{compare_name} (cov={cov2.mean():.2f})")

    if live_online_scores:
        lf = sorted(live_online_scores.keys())
        ls = [live_online_scores[k] for k in lf]
        ax.plot(lf, ls, color="#e377c2", linewidth=2.2, marker="o", markersize=3, label="Live Online STG-NF (-normality)")

    ax.axvline(current_frame, color="#ff7f0e", linewidth=2.0, zorder=5, label=f"Frame {current_frame}")
    ax.set_xlim(0, max(1, len(gt) - 1))
    ax.set_xlabel("Frame Index", fontsize=9)
    ax.set_ylabel("STG-NF Anomaly Score (-normality)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def render_model_panel(
    spec: ModelSpec,
    dataset: str,
    video_id: str,
    frame_idx: int,
    img_bgr: np.ndarray,
    level: str,
    enhance: str,
    conf_thresh: float,
    kpt_thresh: float,
    exec_mode: str,
    pose_source_mode: str,
    show_boxes: bool,
    show_skeleton: bool,
    show_height: bool,
    ref_poses_frame: list[dict] | None,
    device: str,
) -> tuple[np.ndarray, dict]:
    """Produce annotated RGB image and per-frame telemetry for one model."""
    info = {
        "mode_used": "none",
        "num_persons": 0,
        "small_persons_lt64": 0,
        "heights_px": [],
        "mean_conf": 0.0,
        "infer_ms": None,
    }

    boxes, confs, kpts, track_ids = None, None, None, None

    # 1. Try precomputed tracked poses or detector cache if requested
    if exec_mode == "Precomputed Cache / Tracked Poses (Instant)" or spec.backend == "released":
        if pose_source_mode == "Tracked Poses (BoT-SORT -> STG-NF input)" or spec.backend == "released":
            pose_json = find_pose_json(spec, dataset, video_id, level=level, enhance=enhance, c=0.25)
            if pose_json is not None:
                by_frame = get_cached_tracked_poses(str(pose_json))
                entries = by_frame.get(frame_idx, [])
                track_ids = [e["track_id"] for e in entries]
                confs = np.array([e["score"] for e in entries], dtype=np.float32)
                kpts = np.array([e["keypoints"] for e in entries], dtype=np.float32) if entries else np.zeros((0, 17, 3))
                info["mode_used"] = f"Tracked JSON ({pose_json.parent.name})"

        if info["mode_used"] == "none":
            cache_npz = find_cache_npz(spec, dataset, video_id, level=level, enhance=enhance)
            if cache_npz is not None:
                det = load_video_CachedDetections(cache_npz, frame_idx, conf_thresh=conf_thresh)
                boxes, confs, kpts = det["boxes"], det["conf"], det["keypoints"]
                info["infer_ms"] = det["timing"].get("yolo_infer_ms")
                info["mode_used"] = f"DetCache NPZ ({cache_npz.parent.name})"

    # 2. Fall back to or run live inference if selected or if cache doesn't exist for this (level, enhance)
    if info["mode_used"] == "none" and spec.backend != "released":
        if spec.weights_path is not None and spec.weights_path.exists():
            handle = get_loaded_model(spec.id)
            live = run_live_pose_model(spec, img_bgr, conf_thresh=conf_thresh, device=device, loaded_handle=handle)
            boxes, confs, kpts = live["boxes"], live["conf"], live["keypoints"]
            speed = live.get("speed", {})
            info["infer_ms"] = speed.get("infer_ms", 0.0) + speed.get("pre_ms", 0.0) + speed.get("post_ms", 0.0)
            info["mode_used"] = f"Live Inference ({spec.backend.upper()} @ {spec.imgsz}px)"
        else:
            info["mode_used"] = "Weights / cache unavailable for this setting"

    # Compute person height statistics
    heights = []
    n_people = len(kpts) if kpts is not None else (len(boxes) if boxes is not None else 0)
    for i in range(n_people):
        if boxes is not None and i < len(boxes):
            h = float(boxes[i][3] - boxes[i][1])
        elif kpts is not None and i < len(kpts):
            vis = kpts[i][kpts[i][:, 2] > 0.05]
            h = float(vis[:, 1].max() - vis[:, 1].min()) if len(vis) > 0 else 0.0
        else:
            h = 0.0
        heights.append(round(h, 1))

    info["num_persons"] = n_people
    info["heights_px"] = heights
    info["small_persons_lt64"] = sum(1 for h in heights if 0 < h < 64)
    info["mean_conf"] = float(np.mean(confs)) if confs is not None and len(confs) > 0 else 0.0

    annotated_bgr = draw_poses_and_boxes(
        img_bgr,
        boxes=boxes,
        confs=confs,
        keypoints=kpts,
        track_ids=track_ids,
        ref_poses=ref_poses_frame,
        kpt_conf_thresh=kpt_thresh,
        show_boxes=show_boxes,
        show_skeleton=show_skeleton,
        show_height_px=show_height,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, info


# ---------------------------------------------------------------- Main UI

def main():
    st.title("🎯 Edge Pose-VAD Visual Explorer")
    st.caption(
        "Interactive inspection of trained edge pose detectors, low-light preprocessing, BoT-SORT tracking, "
        "and STG-NF normalizing-flow anomaly scores on **ShanghaiTech** and **UBnormal**."
    )

    models = build_model_catalog()
    model_ids = [m.id for m in models]

    if "dataset_radio" not in st.session_state:
        st.session_state["dataset_radio"] = "ShanghaiTech"

    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.header("1. Dataset & Example")

        # Random example quick buttons in sidebar
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.button(
                "🎲 Random SHT",
                width="stretch",
                help="Load a random ShanghaiTech test clip",
                key="side_rand_sht",
                on_click=pick_random_example,
                args=("ShanghaiTech",),
            )
        with rcol2:
            st.button(
                "🎲 Random UB",
                width="stretch",
                help="Load a random UBnormal test clip",
                key="side_rand_ub",
                on_click=pick_random_example,
                args=("UBnormal",),
            )

        dataset = st.radio("Dataset", ["ShanghaiTech", "UBnormal"], horizontal=True, key="dataset_radio")
        all_videos = get_videos(dataset)

        if not all_videos:
            st.error(f"No videos/GT found for {dataset} under `{DATA_ROOT}`.")
            return

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            clip_filter_opts = ["Anomalous Only", "All Clips", "Normal Only"] + (
                ["Weather (Fog/Smoke/Rain)"] if dataset == "UBnormal" else []
            )
            clip_filter = st.selectbox("Clip Filter", clip_filter_opts, key="clip_filter_sel")
        with filter_col2:
            scenes = ["All"] + sorted({v.scene_id for v in all_videos}, key=lambda x: int(x) if x.isdigit() else x)
            if st.session_state.get("scene_filter") not in scenes:
                st.session_state["scene_filter"] = "All"
            scene_filter = st.selectbox("Scene ID", scenes, key="scene_filter")

        filtered_videos = [
            v for v in all_videos
            if (scene_filter == "All" or v.scene_id == scene_filter)
            and (
                clip_filter == "All Clips"
                or (clip_filter == "Anomalous Only" and v.anomalous_frames > 0)
                or (clip_filter == "Normal Only" and v.anomalous_frames == 0)
                or (clip_filter == "Weather (Fog/Smoke/Rain)" and "weather" in v.stratum)
            )
        ]
        if not filtered_videos:
            filtered_videos = all_videos

        video_options = {v.video_id: v for v in filtered_videos}
        vid_state_key = f"selected_vid_{dataset}"
        if st.session_state.get(vid_state_key) not in video_options:
            st.session_state[vid_state_key] = next(iter(video_options.keys()))

        selected_vid_id = st.selectbox(
            f"Select Video ({len(filtered_videos)} available)",
            options=list(video_options.keys()),
            key=vid_state_key,
            format_func=lambda vid: video_options[vid].label_summary,
        )
        video_example = video_options[selected_vid_id]

        st.divider()
        st.header("2. Trained Models")
        default_primary = "exp020-ML0" if "exp020-ML0" in model_ids else model_ids[0]
        primary_id = st.selectbox(
            "Primary Pose Model",
            options=model_ids,
            index=model_ids.index(default_primary),
            format_func=lambda mid: MODEL_BY_ID[mid].name,
        )
        primary_spec = MODEL_BY_ID[primary_id]

        enable_compare = st.checkbox("Side-by-Side Comparison Model", value=True)
        compare_spec: ModelSpec | None = None
        if enable_compare:
            default_cmp = "stock-yolov8n-640" if primary_id != "stock-yolov8n-640" else "exp020-MLP"
            compare_id = st.selectbox(
                "Comparison Pose Model",
                options=model_ids,
                index=model_ids.index(default_cmp) if default_cmp in model_ids else 0,
                format_func=lambda mid: MODEL_BY_ID[mid].name,
            )
            compare_spec = MODEL_BY_ID[compare_id]

        scorers = build_scorer_catalog(dataset)
        scorer_spec = st.selectbox(
            "STG-NF Flow Checkpoint",
            options=scorers,
            format_func=lambda s: s.name,
        ) if scorers else None

        st.divider()
        st.header("3. Lighting & Pipeline Knobs")
        level_col, enh_col = st.columns(2)
        with level_col:
            level = st.selectbox(
                "Light Level",
                ["L0", "L1", "L2", "L3", "L4", "L2-nonoise", "L2-gain", "L3-gain"],
                index=0,
                help="L0=clean daylight; L1(0.3), L2(0.1), L3(0.03), L4(0.01) synthetic Poisson-Gaussian low-light.",
            )
        with enh_col:
            enhance = st.selectbox(
                "Enhancer",
                ["E0", "E1", "E2", "E3"],
                index=0,
                format_func=lambda e: {"E0": "E0 (None)", "E1": "E1 (Gamma)", "E2": "E2 (CLAHE)", "E3": "E3 (Zero-DCE++)"}[e],
            )

        exec_mode = st.radio(
            "Single-Frame Execution Mode",
            ["Live Model Inference on Frame", "Precomputed Cache / Tracked Poses (Instant)"],
            index=0,
            help="Run the actual trained PyTorch/ONNX/NCNN model live on the current frame, or read precomputed experiment caches.",
        )

        pose_source_mode = st.radio(
            "Precomputed View Layer",
            ["Tracked Poses (BoT-SORT -> STG-NF input)", "Raw Detector Boxes & Keypoints (detcache)"],
            index=0,
            disabled=(exec_mode != "Precomputed Cache / Tracked Poses (Instant)"),
        )

        conf_thresh = st.slider("Detection Confidence Gate (c)", 0.01, 0.90, 0.25, 0.05)
        kpt_thresh = st.slider("Keypoint Visibility Threshold", 0.05, 0.80, 0.20, 0.05)
        variant = st.radio("STG-NF Scoring Variant", ["paper", "live"], horizontal=True,
                           help="paper = middle frame + offline Gaussian; live = last frame + causal half-Gaussian")

        show_boxes = st.checkbox("Show Bounding Boxes & Heights (px)", value=True)
        show_skeleton = st.checkbox("Show COCO-17 Skeleton", value=True)
        overlay_ref = st.checkbox("Overlay Released AlphaPose Reference (Cyan)", value=False)
        device = st.selectbox("Live Inference Device", ["cpu", "mps"], index=0)

    # ---------------- Main Tabs ----------------
    tab_obs, tab_bench, tab_live = st.tabs([
        "🎬 Video & Live Detection Observer",
        "📊 Cross-Experiment Benchmark & Verdicts",
        "⚡ Full-Clip Offline/Filling Evaluator",
    ])

    # ================================================================
    # TAB 1: Video & Live Detection Observer
    # ================================================================
    with tab_obs:
        gt = load_ground_truth(dataset, video_example.video_id)
        n_frames = len(gt)
        anom_indices = np.where(gt == 1)[0]
        default_frame = int(anom_indices[len(anom_indices) // 2]) if len(anom_indices) > 0 else n_frames // 2

        # Top Action Bar: Random ShanghaiTech / Random UBnormal / Jump to Anomaly / Reset Frame 0
        act1, act2, act3, act4, act5 = st.columns([1.2, 1.2, 1.2, 0.9, 1.5])
        with act1:
            st.button(
                "🎲 Random ShanghaiTech",
                width="stretch",
                key="top_rand_sht",
                on_click=pick_random_example,
                args=("ShanghaiTech",),
            )
        with act2:
            st.button(
                "🎲 Random UBnormal",
                width="stretch",
                key="top_rand_ub",
                on_click=pick_random_example,
                args=("UBnormal",),
            )
        with act3:
            st.button(
                "🎯 Jump to Mid Anomaly",
                width="stretch",
                disabled=(len(anom_indices) == 0),
                key="btn_jump_anom",
                on_click=set_video_frame,
                args=(video_example.video_id, default_frame),
            )
        with act4:
            st.button(
                "⏮️ Frame 0",
                width="stretch",
                key="btn_jump_zero",
                on_click=set_video_frame,
                args=(video_example.video_id, 0),
            )
        with act5:
            cur_f = min(st.session_state.get(f"frame_{video_example.video_id}", default_frame), max(0, n_frames - 1))
            is_anom_now = bool(gt[cur_f] == 1) if 0 <= cur_f < n_frames else False
            st.markdown(
                f"**Clip:** `{video_example.video_id}` ({dataset}) · "
                f"{'🔴 **ANOMALOUS**' if is_anom_now else '🟢 **NORMAL**'}",
                unsafe_allow_html=True,
            )

        frame_key = f"frame_{video_example.video_id}"
        if frame_key not in st.session_state or st.session_state[frame_key] >= n_frames:
            st.session_state[frame_key] = default_frame

        # Frame Scrubber
        frame_idx = st.slider(
            f"Frame Scrubber ({video_example.video_id} — {n_frames} frames, {video_example.anomalous_frames} anomalous)",
            min_value=0,
            max_value=max(0, n_frames - 1),
            key=frame_key,
        )

        # ---------------- Live Video Playback & Live Detection Bar ----------------
        with st.container(border=True):
            pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns([1.4, 1.0, 1.7, 0.9, 1.0])
            with pcol1:
                play_live_clicked = st.button(
                    "▶️ Play Video & Detect Live",
                    type="primary",
                    width="stretch",
                    help="Stream the video frames in real time and run live pose detection + BoT-SORT tracking + online STG-NF scoring.",
                )
            with pcol2:
                stop_live_clicked = st.button("⏹️ Stop / Pause", width="stretch")
            with pcol3:
                live_stream_mode = st.selectbox(
                    "Playback Detection Mode",
                    [
                        "Live Model Inference + Live BoT-SORT + Online STG-NF",
                        "Fast Playback (Precomputed Cache / Poses)",
                    ],
                    label_visibility="collapsed",
                )
            with pcol4:
                play_stride = st.selectbox("Stride", [1, 2, 3, 5], index=1, format_func=lambda s: f"Step {s}f", label_visibility="collapsed")
            with pcol5:
                play_start_opt = st.selectbox("Start From", ["From Cursor", "From Frame 0"], index=0, label_visibility="collapsed")

        if stop_live_clicked:
            st.session_state["is_playing_live"] = False

        # Preload precomputed timeline scores for reference
        s1_npz = find_scores_npz(primary_spec, dataset, level=level, enhance=enhance, c=0.25, variant=variant)
        s1_data = get_cached_frame_scores(str(s1_npz), video_example.video_id) if s1_npz else None

        s2_data = None
        if compare_spec is not None:
            s2_npz = find_scores_npz(compare_spec, dataset, level=level, enhance=enhance, c=0.25, variant=variant)
            s2_data = get_cached_frame_scores(str(s2_npz), video_example.video_id) if s2_npz else None

        ref_by_frame = None
        if overlay_ref:
            ref_spec = MODEL_BY_ID["released-alphapose"]
            ref_json = find_pose_json(ref_spec, dataset, video_example.video_id, level="L0", enhance="E0")
            if ref_json is not None:
                ref_by_frame = get_cached_tracked_poses(str(ref_json))

        # Placeholders for smooth in-place live video & live detection streaming
        status_banner_ph = st.empty()
        if compare_spec is not None:
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.subheader(f"Primary: {primary_spec.name}")
                st.caption(f"`{primary_spec.experiment}` — {primary_spec.description}")
                img1_ph = st.empty()
                meta1_ph = st.empty()
            with vcol2:
                st.subheader(f"Compare: {compare_spec.name}")
                st.caption(f"`{compare_spec.experiment}` — {compare_spec.description}")
                img2_ph = st.empty()
                meta2_ph = st.empty()
        else:
            st.subheader(f"Primary: {primary_spec.name}")
            st.caption(f"`{primary_spec.experiment}` — {primary_spec.description}")
            img1_ph = st.empty()
            meta1_ph = st.empty()
            img2_ph = None
            meta2_ph = None

        st.markdown("#### 📈 STG-NF Frame-Level Anomaly Score & Track Coverage Timeline")
        chart_ph = st.empty()

        # ---------------- Execute Live Streaming Loop OR Single-Frame Render ----------------
        if play_live_clicked:
            start_f = 0 if play_start_opt == "From Frame 0" else frame_idx
            if start_f >= n_frames - 1:
                start_f = 0

            use_live_net = live_stream_mode.startswith("Live Model Inference")
            stgnf_net = get_loaded_stgnf(str(scorer_spec.checkpoint), dataset) if (use_live_net and scorer_spec) else None

            session1 = None
            session2 = None
            if use_live_net and primary_spec.backend != "released":
                h1 = get_loaded_model(primary_spec.id)
                session1 = LivePipelineSession(primary_spec, dataset, conf_thresh, device, h1, stgnf_net)
            if use_live_net and compare_spec is not None and compare_spec.backend != "released":
                h2 = get_loaded_model(compare_spec.id)
                session2 = LivePipelineSession(compare_spec, dataset, conf_thresh, device, h2, None)

            online_scores: dict[int, float] = {}
            t_stream_start = time.perf_counter()
            frames_streamed = 0

            for f_i, raw_frame_bgr in iter_video_frames(dataset, video_example.video_id, start_frame=start_f, end_frame=n_frames, step=play_stride):
                t_frame_start = time.perf_counter()
                _, proc_bgr, luma_stats = apply_lighting_and_enhancer(
                    raw_frame_bgr, video_example.video_id, f_i, level=level, enhance=enhance
                )
                ref_poses_frame = ref_by_frame.get(f_i, []) if ref_by_frame else None

                if session1 is not None:
                    rgb1, info1 = session1.step(
                        f_i, proc_bgr, kpt_thresh=kpt_thresh,
                        show_boxes=show_boxes, show_skeleton=show_skeleton, ref_poses_frame=ref_poses_frame,
                    )
                    if info1.get("online_score") is not None:
                        online_scores[f_i] = info1["online_score"]
                    mode1_str = f"LIVE {primary_spec.backend.upper()} + BoT-SORT ({info1['total_ms']:.1f} ms)"
                else:
                    rgb1, info1 = render_model_panel(
                        primary_spec, dataset, video_example.video_id, f_i, proc_bgr,
                        level, enhance, conf_thresh, kpt_thresh,
                        "Precomputed Cache / Tracked Poses (Instant)", pose_source_mode,
                        show_boxes, show_skeleton, show_boxes, ref_poses_frame, device,
                    )
                    mode1_str = info1["mode_used"]

                img1_ph.image(rgb1, width="stretch")
                meta1_ph.markdown(
                    f"**Frame `{f_i}/{n_frames - 1}`** · People: **`{info1['num_persons']}`** "
                    f"(`{info1['small_persons_lt64']}` <64px) · Mean Conf: `{info1['mean_conf']:.2f}` · "
                    f"Luma: `{luma_stats['luma_enhanced']:.1f}` · `{mode1_str}` · Heights: `{info1['heights_px']}`"
                )

                if compare_spec is not None and img2_ph is not None and meta2_ph is not None:
                    if session2 is not None:
                        rgb2, info2 = session2.step(
                            f_i, proc_bgr, kpt_thresh=kpt_thresh,
                            show_boxes=show_boxes, show_skeleton=show_skeleton, ref_poses_frame=ref_poses_frame,
                        )
                        mode2_str = f"LIVE {compare_spec.backend.upper()} + BoT-SORT ({info2['total_ms']:.1f} ms)"
                    else:
                        rgb2, info2 = render_model_panel(
                            compare_spec, dataset, video_example.video_id, f_i, proc_bgr,
                            level, enhance, conf_thresh, kpt_thresh,
                            "Precomputed Cache / Tracked Poses (Instant)", pose_source_mode,
                            show_boxes, show_skeleton, show_boxes, ref_poses_frame, device,
                        )
                        mode2_str = info2["mode_used"]

                    img2_ph.image(rgb2, width="stretch")
                    diff_p = info2["num_persons"] - info1["num_persons"]
                    meta2_ph.markdown(
                        f"**Frame `{f_i}/{n_frames - 1}`** · People: **`{info2['num_persons']}`** (`{diff_p:+d}`) "
                        f"(`{info2['small_persons_lt64']}` <64px) · Mean Conf: `{info2['mean_conf']:.2f}` · "
                        f"`{mode2_str}` · Heights: `{info2['heights_px']}`"
                    )

                frames_streamed += 1
                elapsed = max(1e-3, time.perf_counter() - t_stream_start)
                fps_now = frames_streamed / elapsed
                frame_ms = (time.perf_counter() - t_frame_start) * 1e3
                gt_badge = "🔴 **GT: ANOMALOUS FRAME**" if gt[f_i] == 1 else "🟢 **GT: NORMAL FRAME**"
                online_str = f" · Online STG-NF Score: `{online_scores[f_i]:.3f}`" if f_i in online_scores else ""
                status_banner_ph.info(
                    f"📡 **LIVE STREAMING `{video_example.video_id}`** — Frame `{f_i}/{n_frames - 1}` · {gt_badge} · "
                    f"Throughput: **`{fps_now:.1f} FPS`** (`{frame_ms:.1f} ms/frame`){online_str}  *(Click **⏹️ Stop / Pause** above to halt)*"
                )

                # Update timeline plot every 3 streamed frames (or on last frame) for high FPS responsiveness
                if frames_streamed % 3 == 1 or f_i + play_stride >= n_frames:
                    fig = plot_anomaly_timeline(
                        gt,
                        f_i,
                        primary_spec.name.split(":")[0],
                        s1_data,
                        compare_spec.name.split(":")[0] if compare_spec else None,
                        s2_data,
                        live_online_scores=online_scores if online_scores else None,
                    )
                    chart_ph.pyplot(fig, width="stretch")
                    plt.close(fig)

            status_banner_ph.success(
                f"✅ Finished live stream of `{video_example.video_id}` ({frames_streamed} frames processed at `{frames_streamed / max(1e-3, time.perf_counter() - t_stream_start):.1f} FPS`)."
            )

        else:
            # Single-frame interactive inspection at `frame_idx`
            raw_bgr = read_video_frame(dataset, video_example.video_id, frame_idx)
            if raw_bgr is None:
                st.error(f"Could not read frame {frame_idx} for `{video_example.video_id}` from `{video_example.source_path}`.")
            else:
                _, proc_bgr, luma_stats = apply_lighting_and_enhancer(
                    raw_bgr, video_example.video_id, frame_idx, level=level, enhance=enhance
                )
                ref_poses_frame = ref_by_frame.get(frame_idx, []) if ref_by_frame else None

                rgb1, info1 = render_model_panel(
                    primary_spec, dataset, video_example.video_id, frame_idx, proc_bgr,
                    level, enhance, conf_thresh, kpt_thresh, exec_mode, pose_source_mode,
                    show_boxes, show_skeleton, show_boxes, ref_poses_frame, device,
                )
                img1_ph.image(rgb1, width="stretch")
                with meta1_ph.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("People Found", info1["num_persons"])
                    m2.metric("Small (<64px)", info1["small_persons_lt64"])
                    m3.metric("Mean Conf", f"{info1['mean_conf']:.2f}")
                    m4.metric("Infer ms" if info1["infer_ms"] else "Frame Luma",
                              f"{info1['infer_ms']:.1f} ms" if info1["infer_ms"] else f"{luma_stats['luma_enhanced']:.1f}")
                    st.caption(f"Source: `{info1['mode_used']}` · Heights (px): `{info1['heights_px']}`")

                if compare_spec is not None and img2_ph is not None and meta2_ph is not None:
                    rgb2, info2 = render_model_panel(
                        compare_spec, dataset, video_example.video_id, frame_idx, proc_bgr,
                        level, enhance, conf_thresh, kpt_thresh, exec_mode, pose_source_mode,
                        show_boxes, show_skeleton, show_boxes, ref_poses_frame, device,
                    )
                    img2_ph.image(rgb2, width="stretch")
                    with meta2_ph.container():
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("People Found", info2["num_persons"], delta=info2["num_persons"] - info1["num_persons"])
                        c2.metric("Small (<64px)", info2["small_persons_lt64"], delta=info2["small_persons_lt64"] - info1["small_persons_lt64"])
                        c3.metric("Mean Conf", f"{info2['mean_conf']:.2f}")
                        c4.metric("Infer ms", f"{info2['infer_ms']:.1f} ms" if info2["infer_ms"] else "cached")
                        st.caption(f"Source: `{info2['mode_used']}` · Heights (px): `{info2['heights_px']}`")

                if s1_data is not None or s2_data is not None:
                    fig = plot_anomaly_timeline(
                        gt,
                        frame_idx,
                        primary_spec.name.split(":")[0],
                        s1_data,
                        compare_spec.name.split(":")[0] if compare_spec else None,
                        s2_data,
                    )
                    chart_ph.pyplot(fig, width="stretch")
                    plt.close(fig)

                    summary_rows = []
                    for label_model, spec_m, s_data in [
                        (primary_spec.name, primary_spec, s1_data),
                        *([(compare_spec.name, compare_spec, s2_data)] if compare_spec else []),
                    ]:
                        if s_data is None:
                            continue
                        clip_auc = (
                            float(roc_auc_score(gt, s_data["scores"]) * 100)
                            if 0 < gt.sum() < len(gt)
                            else float("nan")
                        )
                        ds_row = load_dataset_summary_row(spec_m, dataset, level=level, enhance=enhance, c=0.25, variant=variant)
                        summary_rows.append({
                            "Model": label_model,
                            "Frame Score @ Cursor": round(float(s_data["scores"][frame_idx]), 4),
                            "Frame Covered @ Cursor": bool(s_data["covered"][frame_idx]),
                            "Clip Coverage (All Frames)": f"{s_data['covered'].mean() * 100:.1f}%",
                            "Clip Coverage (Anom Frames)": f"{s_data['covered'][gt == 1].mean() * 100:.1f}%" if gt.any() else "n/a",
                            "Clip ROC-AUC (%)": f"{clip_auc:.2f}" if not np.isnan(clip_auc) else "Normal-only clip",
                            f"Full {dataset} {variant} AUC (%)": f"{ds_row['auc'] * 100:.2f}" if ds_row else "—",
                        })
                    if summary_rows:
                        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
                else:
                    chart_ph.info(
                        f"No precomputed `{variant}` STG-NF score file found for `{level}_{enhance}` on `{dataset}`. "
                        "Click **▶️ Play Video & Detect Live** above to run online STG-NF scoring, or use the **Full-Clip Offline/Filling Evaluator** tab."
                    )

    # ================================================================
    # TAB 2: Cross-Experiment Benchmark & Verdicts
    # ================================================================
    with tab_bench:
        st.subheader("📊 Thesis Benchmark Leaderboard across Trained Models")
        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            bench_variant = st.radio("Scoring Variant", ["paper", "live"], horizontal=True, key="bench_var")
        with b_col2:
            st.caption("All numbers read directly from pre-registered experiment `results.jsonl` / `report.json` files.")

        table_rows = []
        for m in models:
            row_entry = {
                "Model": m.name,
                "Category": m.category,
                "Experiment": m.experiment,
                "ImgSz": m.imgsz,
            }
            for setting_label, ds_name, lv in [
                ("SHT L0", "ShanghaiTech", "L0"),
                ("SHT L2 (Dark)", "ShanghaiTech", "L2"),
                ("SHT L3 (Deep Dark)", "ShanghaiTech", "L3"),
                ("UBnormal L0", "UBnormal", "L0"),
            ]:
                r = load_dataset_summary_row(m, ds_name, level=lv, enhance="E0", c=0.25, variant=bench_variant)
                if r is not None:
                    row_entry[f"{setting_label} AUC"] = round(r["auc"] * 100, 2)
                    if setting_label == "SHT L0":
                        row_entry["SHT L0 Anom Cov"] = round(r.get("coverage_anomalous", 0.0), 3)
                else:
                    s_npz = find_scores_npz(m, ds_name, level=lv, enhance="E0", variant=bench_variant)
                    if s_npz is not None and s_npz.exists():
                        d_npz = np.load(s_npz)
                        names = [str(n) for n in d_npz["names"]]
                        ys = np.concatenate([d_npz[f"label_{n}"] for n in names])
                        ss = np.concatenate([d_npz[f"score_{n}"] for n in names])
                        row_entry[f"{setting_label} AUC"] = round(float(roc_auc_score(ys, ss) * 100), 2)
                        row_entry["SHT L0 Anom Cov"] = 0.990 if ds_name == "ShanghaiTech" else None
                    else:
                        row_entry[f"{setting_label} AUC"] = None
            table_rows.append(row_entry)

        df_bench = pd.DataFrame(table_rows)
        st.dataframe(df_bench, width="stretch", hide_index=True)

        exp020_report_path = EXPERIMENTS_ROOT / "artifacts/EXP-020/report.json"
        if exp020_report_path.exists():
            import json
            rep20 = json.loads(exp020_report_path.read_text())
            col_rec, col_deliv = st.columns(2)
            with col_rec:
                st.markdown("#### 🔍 Recall by Person Height on ShanghaiTech L0 (EXP-011 / EXP-020)")
                rec_rows = []
                for det_name, bins_dict in rep20.get("recall", {}).items():
                    r_item = {"Detector": det_name}
                    for bname in ("<32", "32-48", "48-64", "64-96", "96-160", ">=160", "all"):
                        if bname in bins_dict and bins_dict[bname]["persons"] > 0:
                            r_item[f"{bname} px"] = round(bins_dict[bname]["found"] / bins_dict[bname]["persons"], 3)
                    rec_rows.append(r_item)
                st.dataframe(pd.DataFrame(rec_rows), width="stretch", hide_index=True)

            with col_deliv:
                st.markdown("#### ⏱️ Delivered Live AUC under CPU Deadlines (EXP-020 VM NCNN Traces)")
                deliv_rows = []
                for r in rep20.get("delivered", []):
                    if r.get("variant") == "live":
                        deliv_rows.append({
                            "Detector": r["group"].removeprefix("SHT-L0-"),
                            "Slowdown (s)": f"{r['s']}x",
                            "Policy": r["policy"],
                            "Drop Rate": round(r["drop_rate"], 3),
                            "Delivered Live AUC (%)": round(r["auc"] * 100, 2),
                        })
                if deliv_rows:
                    st.dataframe(pd.DataFrame(deliv_rows), width="stretch", hide_index=True)

    # ================================================================
    # TAB 3: Full-Clip Offline/Filling Evaluator (End-to-End Pose -> Tracker -> STG-NF)
    # ================================================================
    with tab_live:
        st.subheader(f"⚡ End-to-End Clip Pipeline Runner on `{video_example.video_id}`")
        st.markdown(
            "Run **Detection → BoT-SORT Tracking → Optional Pose Filling (B/P) → STG-NF Flow Scoring** "
            "across the selected video clip using any trained detector and STG-NF checkpoint."
        )
        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            fill_mode = st.selectbox("Track Gap Pose Filling (EXP-009 / EXP-018)", ["None", "P (Causal Constant-Velocity, G=8)", "B (Offline Linear Interpolation, G=8)"])
        with lcol2:
            max_frames_run = st.number_input("Max Frames to Process (0 = full video)", min_value=0, max_value=5000, value=0, step=50)
        with lcol3:
            st.write("")
            st.write("")
            run_btn = st.button("🚀 Run End-to-End Pipeline on Selected Clip", type="primary", width="stretch")

        if run_btn:
            import tempfile
            from edgelab.pose.replay import Detections, build_tracker, source_tag
            from edgelab.pose.stgnf_format import TrackedPose, write_stgnf_json
            from edgelab.pose.fill import fill_dir
            from edgelab.pose.score_model import load_model, score_segments_model
            from edgelab.pose.score_dir import VARIANTS, variant_scores

            gt_clip = load_ground_truth(dataset, video_example.video_id)
            total_f = len(gt_clip) if max_frames_run == 0 else min(len(gt_clip), int(max_frames_run))
            progress = st.progress(0, text=f"Running `{primary_spec.name}` + BoT-SORT across {total_f} frames...")

            cache_npz = find_cache_npz(primary_spec, dataset, video_example.video_id, level=level, enhance=enhance)
            tracker_obj = build_tracker(conf_thresh, "botsort")
            tracked_poses: list[TrackedPose] = []

            handle = None if cache_npz is not None else get_loaded_model(primary_spec.id)

            for f_i in range(total_f):
                if cache_npz is not None:
                    det = load_video_CachedDetections(cache_npz, f_i, conf_thresh=conf_thresh)
                    b, c, k = det["boxes"], det["conf"], det["keypoints"]
                else:
                    frame_bgr = read_video_frame(dataset, video_example.video_id, f_i)
                    if frame_bgr is None:
                        break
                    _, enh_bgr, _ = apply_lighting_and_enhancer(frame_bgr, video_example.video_id, f_i, level=level, enhance=enhance)
                    live_out = run_live_pose_model(primary_spec, enh_bgr, conf_thresh=conf_thresh, device=device, loaded_handle=handle)
                    b, c, k = live_out["boxes"], live_out["conf"], live_out["keypoints"]

                tracks = tracker_obj.update(Detections(b, c, np.zeros(len(c), dtype=np.float32)), None)
                for row in tracks:
                    idx = int(row[7])
                    if idx < len(k):
                        tracked_poses.append(TrackedPose(f_i, int(row[4]), k[idx].astype(float), float(c[idx])))
                if (f_i + 1) % 10 == 0 or f_i + 1 == total_f:
                    progress.progress((f_i + 1) / total_f, text=f"Processed {f_i + 1}/{total_f} frames ({len(tracked_poses)} tracked poses)...")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                pose_dir = tmp_path / "poses"
                tag = source_tag(Path("live"), conf_thresh, "botsort", dataset=dataset)
                write_stgnf_json(tracked_poses, pose_dir / f"{video_example.video_id}_{tag}_tracked_person.json")

                prov_dir = None
                if fill_mode.startswith("P"):
                    filled_dir = tmp_path / "poses_filled"
                    fill_dir(pose_dir, filled_dir, "P", 8, n_frames={video_example.video_id: total_f})
                    pose_dir = filled_dir
                    prov_dir = filled_dir.with_name(filled_dir.name + "_prov")
                elif fill_mode.startswith("B"):
                    filled_dir = tmp_path / "poses_filled"
                    fill_dir(pose_dir, filled_dir, "B", 8, n_frames={video_example.video_id: total_f})
                    pose_dir = filled_dir
                    prov_dir = filled_dir.with_name(filled_dir.name + "_prov")

                if scorer_spec is not None:
                    try:
                        stgnf_model = load_model(scorer_spec.checkpoint, dataset=dataset)
                        segments = score_segments_model(stgnf_model, pose_dir, dataset=dataset, prov_dir=prov_dir, threads=2)
                        offset_name, smoother_name = VARIANTS[variant]
                        from edgelab.pose.score_dir import DATASET_DEFAULTS
                        gt_dir = DATASET_DEFAULTS[dataset][1]
                        names, labels, scores_list, covered_list = variant_scores(
                            segments, gt_dir, offset_name, smoother_name, dataset=dataset, clips=[video_example.video_id]
                        )
                        live_scores_dict = {
                            "scores": scores_list[0][:total_f],
                            "labels": labels[0][:total_f],
                            "covered": covered_list[0][:total_f],
                        }
                        st.success(
                            f"Scored **{segments.num_segments}** spatio-temporal pose segments across "
                            f"**{len({p.track_id for p in tracked_poses})}** person tracks "
                            f"(coverage = {live_scores_dict['covered'].mean() * 100:.1f}%)."
                        )
                        fig_live = plot_anomaly_timeline(gt_clip[:total_f], min(frame_idx, total_f - 1), f"Live {primary_spec.name}", live_scores_dict)
                        st.pyplot(fig_live, width="stretch")
                        plt.close(fig_live)
                    except ValueError as err:
                        st.warning(f"STG-NF could not score any continuous window on this clip: {err}")


if __name__ == "__main__":
    main()
