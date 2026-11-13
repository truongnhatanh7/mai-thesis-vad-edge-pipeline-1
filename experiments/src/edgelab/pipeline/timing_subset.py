"""Timing subset (EXP-004 plan §3.1): 2 test videos for each of the 6 scenes with the most test videos, first 240 frames.

    uv run python -m edgelab.pipeline.timing_subset --frames-root <testing/frames> [--copy-to <vm-exchange/frames>]

Choice rules, fixed before any timing run: scenes ranked by number of test videos (ties -> lower scene id); within a
scene, videos shorter than 240 frames are not eligible; the 2 videos are drawn with random.Random(f"{seed}|{scene}").
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

N_SCENES, PER_SCENE, N_FRAMES, SEED = 6, 2, 240, 0
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-004-os-deadlines"


def select_subset(frame_counts: dict[str, int], seed: int = SEED, n_scenes: int = N_SCENES,
                  per_scene: int = PER_SCENE, n_frames: int = N_FRAMES) -> list[str]:
    """frame_counts: video name '<scene>_<clip>' -> number of frames. Returns the chosen video names, sorted."""
    per_scene_count = Counter(video.split("_")[0] for video in frame_counts)
    top_scenes = sorted(per_scene_count, key=lambda scene: (-per_scene_count[scene], scene))[:n_scenes]
    chosen = []
    for scene in top_scenes:
        eligible = sorted(v for v, n in frame_counts.items() if v.split("_")[0] == scene and n >= n_frames)
        chosen += random.Random(f"{seed}|{scene}").sample(eligible, per_scene)
    return sorted(chosen)


def copy_frames(videos: list[str], frames_root: Path, dest: Path, n_frames: int = N_FRAMES) -> int:
    """Copy the first n_frames jpgs of each video to dest/<video>/ (the VM cannot see the host's frames)."""
    copied = 0
    for video in videos:
        (dest / video).mkdir(parents=True, exist_ok=True)
        for jpg in sorted((frames_root / video).glob("*.jpg"))[:n_frames]:
            shutil.copy2(jpg, dest / video / jpg.name)
            copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "timing_subset.txt")
    parser.add_argument("--copy-to", type=Path, default=None, help="also copy the subset's frames here")
    args = parser.parse_args()
    counts = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(args.frames_root.iterdir()) if d.is_dir()}
    videos = select_subset(counts)
    args.out.write_text("\n".join(videos) + "\n")
    print(f"{len(videos)} videos -> {args.out}: {' '.join(videos)}")
    if args.copy_to:
        print(f"copied {copy_frames(videos, args.frames_root, args.copy_to)} frames to {args.copy_to}")


if __name__ == "__main__":
    main()
