"""EXP-020: COCO 2017 person-keypoints -> two Ultralytics pose datasets over the SAME images.

    uv run --extra pose python -m edgelab.pose.coco_pose_labels --coco ~/thesis-data/coco --out ~/thesis-data/exp020/data

Images = every image with at least one non-crowd person (box width and height > 0). Same train2017 / val2017 split.
  standard : exactly the Ultralytics converter (ultralytics.data.converter.convert_coco, use_keypoints=True): a person
             with no labelled keypoint (num_keypoints == 0, COCO: area < 32^2 or crowded) is NOT written, so its pixels
             are background in training. An image whose persons all lack keypoints gets an empty label file.
  partial  : every non-crowd person is written. A person without a labelled keypoint gets its box and 17 keypoints
             (x, y, v) = (0, 0, 0); the training loss then ignores its keypoints (edgelab.pose.loss_patch).
Label row = `0 cx cy w h x1 y1 v1 ... x17 y17 v17`, coordinates divided by image width / height, written with "%g"
exactly like the converter (rows are de-duplicated like the converter does).
"""

import argparse
import json
from pathlib import Path

import numpy as np

HEIGHT_EDGES = [0, 32, 48, 64, 96, 160, np.inf]  # px of the COCO box height; same bins as EXP-011
HEIGHT_NAMES = ["<32", "32-48", "48-64", "64-96", "96-160", ">=160"]
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
SPLITS = {"train2017": "person_keypoints_train2017.json", "val2017": "person_keypoints_val2017.json"}


def has_keypoints(ann: dict) -> bool:
    """The converter's rule: at least one keypoint with a non-zero visibility flag."""
    return any((ann.get("keypoints") or [])[2::3])


def person_row(ann: dict, width: int, height: int, partial: bool) -> list[float] | None:
    """One COCO annotation -> label row (list of floats), or None if the person is not written.

    Crowd regions and boxes without area are never written. Keypoint-less persons are written only when `partial`.
    """
    if ann.get("iscrowd", False):
        return None
    box = np.array(ann["bbox"], dtype=np.float64)  # COCO: top-left x, y, width, height
    box[:2] += box[2:] / 2  # -> centre
    box[[0, 2]] /= width
    box[[1, 3]] /= height
    if box[2] <= 0 or box[3] <= 0:
        return None
    row = [0.0, *box.tolist()]  # class 0 = person
    if has_keypoints(ann):
        kpts = np.array(ann["keypoints"]).reshape(-1, 3) / np.array([width, height, 1])
        return row + kpts.reshape(-1).tolist()
    if partial:
        return row + [0.0] * 51
    return None


def format_rows(rows: list[list[float]]) -> str:
    """Converter formatting: "%g" per number, duplicate rows dropped (first occurrence kept)."""
    return "".join(("%g " * len(r)).rstrip() % tuple(r) + "\n" for r in dict.fromkeys(map(tuple, rows)))


def height_bin(box_height: float) -> int:
    return int(np.searchsorted(HEIGHT_EDGES, box_height, side="right") - 1)


def person_images(data: dict) -> dict[int, list[dict]]:
    """image id -> its person annotations, only images with at least one non-crowd person with a box of positive size."""
    by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    images = {img["id"]: img for img in data["images"]}
    keep = {}
    for image_id, anns in by_image.items():
        img = images[image_id]
        if any(person_row(a, img["width"], img["height"], partial=True) is not None for a in anns):
            keep[image_id] = anns
    return keep


def build_split(data: dict, labels_dir: Path, partial: bool) -> dict:
    """Write one label file per person image. Returns counts: images, persons by height bin, persons with keypoints."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = {img["id"]: img for img in data["images"]}
    persons = np.zeros(len(HEIGHT_NAMES), dtype=np.int64)
    with_kp = np.zeros(len(HEIGHT_NAMES), dtype=np.int64)
    n_images, empty_images, names = 0, 0, []
    for image_id, anns in sorted(person_images(data).items()):
        img = images[image_id]
        rows = []
        for ann in anns:
            row = person_row(ann, img["width"], img["height"], partial)
            if row is None:
                continue
            rows.append(row)
        unique = list(dict.fromkeys(map(tuple, rows)))  # counts follow what is written (the converter de-duplicates)
        for r in unique:
            b = height_bin(r[4] * img["height"])
            persons[b] += 1
            with_kp[b] += int(any(r[7::3]))
        name = Path(img["file_name"]).stem
        (labels_dir / f"{name}.txt").write_text(format_rows(rows))
        names.append(img["file_name"])
        n_images += 1
        empty_images += int(not unique)
    return {"images": n_images, "empty_label_images": empty_images, "persons": int(persons.sum()),
            "persons_with_keypoints": int(with_kp.sum()),
            "persons_by_height": dict(zip(HEIGHT_NAMES, map(int, persons))),
            "persons_with_keypoints_by_height": dict(zip(HEIGHT_NAMES, map(int, with_kp))),
            "image_files": names}


def write_dataset(out: Path, images_root: Path, stats_by_split: dict, kind: str):
    """Image lists (absolute paths through the `images` symlink so that /images/ -> /labels/ resolves to this dataset),
    dataset.yaml."""
    root = out / kind
    for split, stats in stats_by_split.items():
        lines = [str(root / "images" / split / name) for name in stats["image_files"]]
        (root / f"{split}.txt").write_text("\n".join(lines) + "\n")
    link = root / "images"
    if not link.exists():
        link.symlink_to(images_root)
    (root / "dataset.yaml").write_text(
        f"path: {root}\ntrain: train2017.txt\nval: val2017.txt\nkpt_shape: [17, 3]\nflip_idx: {FLIP_IDX}\nnames:\n  0: person\n")


def build_all(coco: Path, out: Path) -> dict:
    """Both datasets; the manifest says whether the two image lists are identical (they must be)."""
    manifest = {}
    loaded = {split: json.loads((coco / "annotations" / fname).read_text()) for split, fname in SPLITS.items()}
    files = {}
    for kind, partial in (("standard", False), ("partial", True)):
        stats = {split: build_split(data, out / kind / "labels" / split, partial) for split, data in loaded.items()}
        write_dataset(out, coco / "images", stats, kind)
        files[kind] = {split: s["image_files"] for split, s in stats.items()}
        manifest[kind] = {split: {k: v for k, v in s.items() if k != "image_files"} for split, s in stats.items()}
    manifest["image_lists_equal"] = files["standard"] == files["partial"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, required=True, help="folder with annotations/ and images/{train2017,val2017}")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_all(args.coco, args.out)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
