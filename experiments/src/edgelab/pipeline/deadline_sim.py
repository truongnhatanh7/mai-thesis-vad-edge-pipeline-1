"""Deadline simulation (EXP-004 plan §3.1 step 2): which camera frames does a single-worker, latest-frame system process?

Camera frame i arrives at time i / fps. One worker. When it becomes free it takes the NEWEST frame that has arrived;
older frames that were waiting are dropped. If no new frame has arrived yet, it waits for the next one.
Time is kept in frame units (1.0 = one camera period) so that exact multiples do not suffer from float round-off.
"""

import hashlib

import numpy as np

EPS = 1e-9  # a frame arriving exactly when the worker frees up counts as arrived


def simulate_latest_frame(service_ms: np.ndarray, n_frames: int, fps: float = 24.0) -> np.ndarray:
    """-> keep_mask (n_frames,) bool, True = frame was processed. service_ms[k] = service time of the k-th processed frame."""
    service_frames = np.asarray(service_ms, dtype=float) / 1000.0 * fps
    keep = np.zeros(n_frames, dtype=bool)
    frame, free_at, k = 0, 0.0, 0  # the first frame is processed at t = 0
    while frame < n_frames:
        if k >= len(service_frames):
            raise ValueError(f"service_ms has {len(service_frames)} entries, need more than {k}")
        keep[frame] = True
        free_at = max(free_at, frame) + service_frames[k]  # the worker starts the frame when it has both arrived and the worker is free
        k += 1
        newest_arrived = int(np.floor(free_at + EPS))
        frame = max(frame + 1, newest_arrived)  # nothing newer yet -> wait for the next frame
    return keep


def block_bootstrap_service(trace_ms: np.ndarray, n: int, block: int = 24, seed: int = 0) -> np.ndarray:
    """n service times made of whole blocks of `block` consecutive frames of the trace, drawn with replacement.

    Blocks are the non-overlapping aligned chunks trace[0:block], trace[block:2*block], ...; a tail shorter
    than one block is unused. Blocks keep the short-range correlation of the trace (a slow frame is followed by slow frames).
    """
    trace_ms = np.asarray(trace_ms, dtype=float)
    n_blocks = len(trace_ms) // block
    if n_blocks == 0:
        raise ValueError(f"trace has {len(trace_ms)} frames, need at least one block of {block}")
    blocks = trace_ms[: n_blocks * block].reshape(n_blocks, block)
    rng = np.random.default_rng(seed)
    n_draws = -(-n // block)  # ceil
    return blocks[rng.integers(0, n_blocks, n_draws)].reshape(-1)[:n]


def video_seed(seed: int, video: str) -> int:
    """Stable across runs and machines (Python's built-in hash() is randomised per process)."""
    return int.from_bytes(hashlib.sha256(f"{seed}|{video}".encode()).digest()[:8], "little")


def keep_masks_for_videos(frame_counts: dict[str, int], trace_ms: np.ndarray, fps: float = 24.0,
                          seed: int = 0) -> dict[str, np.ndarray]:
    """One keep_mask per video; each video draws its own service times from the trace with its own stable seed."""
    masks = {}
    for video, n_frames in frame_counts.items():
        service = block_bootstrap_service(trace_ms, n_frames, seed=video_seed(seed, video))
        masks[video] = simulate_latest_frame(service, n_frames, fps)
    return masks
