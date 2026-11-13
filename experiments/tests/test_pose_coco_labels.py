"""EXP-020 labels: standard == Ultralytics' own converter; partial = same rows + box-only rows for keypoint-less persons."""

import json
from pathlib import Path

from edgelab.pose.coco_pose_labels import (HEIGHT_NAMES, build_all, build_split, format_rows, height_bin, person_images,
                                           person_row)


def _kp(n_visible: int) -> list[int]:
    """51 numbers; the first n_visible keypoints are labelled (v = 2) at distinct positions, the rest 0 0 0."""
    out = []
    for i in range(17):
        out += [10 + 3 * i, 20 + 2 * i, 2] if i < n_visible else [0, 0, 0]
    return out


def _data() -> dict:
    cat = {"category_id": 1}
    anns = [
        {"id": 1, "image_id": 1, "bbox": [10, 20, 40, 100], "keypoints": _kp(17), "num_keypoints": 17, "iscrowd": 0, **cat},
        {"id": 2, "image_id": 1, "bbox": [200, 50, 10, 20], "keypoints": _kp(0), "num_keypoints": 0, "iscrowd": 0, **cat},  # small
        {"id": 3, "image_id": 1, "bbox": [300, 50, 50, 50], "keypoints": _kp(0), "num_keypoints": 0, "iscrowd": 1, **cat},  # crowd
        {"id": 4, "image_id": 2, "bbox": [5, 5, 12, 30], "keypoints": _kp(0), "num_keypoints": 0, "iscrowd": 0, **cat},  # only keypoint-less
        {"id": 5, "image_id": 3, "bbox": [0, 0, 50, 50], "keypoints": _kp(0), "num_keypoints": 0, "iscrowd": 1, **cat},  # crowd only
        {"id": 6, "image_id": 4, "bbox": [10, 10, 0, 40], "keypoints": _kp(0), "num_keypoints": 0, "iscrowd": 0, **cat},  # no width
        {"id": 7, "image_id": 1, "bbox": [10, 20, 40, 100], "keypoints": _kp(17), "num_keypoints": 17, "iscrowd": 0, **cat},  # duplicate of 1
    ]
    images = [{"id": i, "file_name": f"{i:012d}.jpg", "width": 400, "height": 200} for i in (1, 2, 3, 4, 5)]
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}


def test_person_images_needs_a_non_crowd_person_with_area():
    assert sorted(person_images(_data())) == [1, 2]  # image 3: crowd only; image 4: zero width; image 5: no annotation


def test_person_row_rules():
    d = _data()
    a = {a["id"]: a for a in d["annotations"]}
    assert person_row(a[3], 400, 200, partial=True) is None  # crowd never written
    assert person_row(a[6], 400, 200, partial=True) is None  # no area
    assert person_row(a[2], 400, 200, partial=False) is None  # keypoint-less dropped in standard
    row = person_row(a[2], 400, 200, partial=True)
    assert row[:5] == [0.0, (200 + 5) / 400, (50 + 10) / 200, 10 / 400, 20 / 200]
    assert len(row) == 5 + 51 and row[5:] == [0.0] * 51  # all 17 keypoints x = y = v = 0
    full = person_row(a[1], 400, 200, partial=False)
    assert full == person_row(a[1], 400, 200, partial=True) and len(full) == 56
    assert full[5:8] == [10 / 400, 20 / 200, 2.0]


def test_partial_is_standard_plus_box_only_rows(tmp_path):
    data = _data()
    s = build_split(data, tmp_path / "std", partial=False)
    p = build_split(data, tmp_path / "par", partial=True)
    assert s["images"] == p["images"] == 2
    assert (s["persons"], p["persons"]) == (1, 3)  # duplicate row collapsed; keypoint-less persons only in partial
    assert s["persons_with_keypoints"] == p["persons_with_keypoints"] == 1
    assert s["empty_label_images"] == 1 and p["empty_label_images"] == 0  # image 2: background in standard
    std1 = (tmp_path / "std" / "000000000001.txt").read_text().splitlines()
    par1 = (tmp_path / "par" / "000000000001.txt").read_text().splitlines()
    assert len(std1) == 1 and len(par1) == 2 and par1[0] == std1[0]
    assert (tmp_path / "std" / "000000000002.txt").read_text() == ""
    assert par1[1].split()[5:] == ["0"] * 51
    # persons by height bin: heights 100, 20 (image 1) and 30 (image 2)
    assert p["persons_by_height"]["<32"] == 2 and p["persons_by_height"]["96-160"] == 1
    assert s["persons_by_height"]["<32"] == 0 and sum(p["persons_by_height"].values()) == 3


def test_height_bins():
    assert [height_bin(h) for h in (0, 31.9, 32, 47, 48, 64, 95, 96, 159, 160, 900)] == [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5]
    assert len(HEIGHT_NAMES) == 6


def test_standard_labels_equal_ultralytics_converter(tmp_path):
    from ultralytics.data.converter import convert_coco

    data = _data()
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir()
    (ann_dir / "person_keypoints_toy.json").write_text(json.dumps(data))
    convert_coco(str(ann_dir), str(tmp_path / "conv"), use_keypoints=True, cls91to80=False)
    theirs = tmp_path / "conv" / "labels" / "person_keypoints_toy"
    build_split(data, tmp_path / "std", partial=False)
    for image_id in (1, 2):  # the person images; the converter also writes images 3 and 4 (crowd / no area) as empty files
        name = f"{image_id:012d}.txt"
        assert (tmp_path / "std" / name).read_text() == (theirs / name).read_text(), name
    assert (theirs / "000000000003.txt").read_text() == ""


def test_build_all_writes_lists_yaml_and_manifest(tmp_path):
    coco = tmp_path / "coco"
    (coco / "annotations").mkdir(parents=True)
    (coco / "images" / "train2017").mkdir(parents=True)
    (coco / "images" / "val2017").mkdir(parents=True)
    for split in ("train", "val"):
        (coco / "annotations" / f"person_keypoints_{split}2017.json").write_text(json.dumps(_data()))
    manifest = build_all(coco, tmp_path / "out")
    assert manifest["image_lists_equal"] is True
    assert manifest["partial"]["train2017"]["persons"] == 3 and manifest["standard"]["val2017"]["persons"] == 1
    listing = (tmp_path / "out" / "partial" / "train2017.txt").read_text().split()
    assert listing[0] == str(tmp_path / "out" / "partial" / "images" / "train2017" / "000000000001.jpg")
    assert (tmp_path / "out" / "partial" / "images").is_symlink()
    yaml = (tmp_path / "out" / "standard" / "dataset.yaml").read_text()
    assert "kpt_shape: [17, 3]" in yaml and "train: train2017.txt" in yaml
    assert Path(yaml.splitlines()[0].split(": ")[1]) == tmp_path / "out" / "standard"


def test_format_rows_dedupes_and_uses_g_format():
    assert format_rows([[0.0, 0.5, 0.25, 0.125, 0.0625], [0.0, 0.5, 0.25, 0.125, 0.0625]]) == "0 0.5 0.25 0.125 0.0625\n"
