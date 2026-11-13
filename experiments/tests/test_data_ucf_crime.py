"""UCF-Crime loader checks. Data-dependent tests skip when the dataset is not on this machine."""


import numpy as np
import pytest

from edgelab.data import ucf_crime as ucf

ROOT = ucf.DEFAULT_ROOT
has_annotations = (ROOT / ucf.ANNOTATION_FILE).exists() and (ROOT / ucf.SPLITS_DIR / "Anomaly_Test.txt").exists()
has_index = (ROOT / "index_test.csv").exists()
needs_annotations = pytest.mark.skipif(not has_annotations, reason="UCF-Crime annotations not downloaded")
needs_index = pytest.mark.skipif(not has_index, reason="index_test.csv not built")
REFERENCE_GT = {"rtfm": ROOT / "reference/RTFM/gt-ucf.npy", "vadclip": ROOT / "reference/VadCLIP/gt_ucf.npy"}


def test_parse_single_and_double_interval_lines():
    one = ucf.parse_annotation_line("Abuse028_x264.mp4  Abuse  165  240  -1  -1  ")
    two = ucf.parse_annotation_line("Arson011_x264.mp4  Arson  150  420  680  1267  ")
    normal = ucf.parse_annotation_line("Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1  ")
    assert (one.name, one.event_class, one.intervals) == ("Abuse028_x264", "Abuse", ((165, 240),))
    assert two.intervals == ((150, 420), (680, 1267))
    assert normal.intervals == () and not normal.is_anomalous


def test_frame_labels_end_inclusive_vs_exclusive():
    video = ucf.AnnotatedVideo("x", "Abuse", ((2, 4),))
    assert ucf.frame_labels(video, 8).tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert ucf.frame_labels(video, 8, end_inclusive=False).tolist() == [0, 0, 1, 1, 0, 0, 0, 0]


def test_frame_labels_clip_interval_past_video_end():
    video = ucf.AnnotatedVideo("x", "Abuse", ((3, 99),))
    assert ucf.frame_labels(video, 5).tolist() == [0, 0, 0, 1, 1]


@needs_annotations
def test_test_list_and_annotations_agree():
    test_list = ucf.load_test_list()
    annotations = ucf.load_annotations()
    assert len(test_list) == 290 and len(annotations) == 290
    assert {ucf.video_name(r) for r in test_list} == set(annotations)
    assert sum(r.startswith("Testing_Normal_Videos_Anomaly/") for r in test_list) == 150
    assert sum(v.is_anomalous for v in annotations.values()) == 140
    assert all(v.intervals for v in annotations.values() if v.is_anomalous)
    assert not any(v.intervals for v in annotations.values() if not v.is_anomalous)


@needs_annotations
def test_train_list_has_1610_videos_disjoint_from_test():
    train = ucf.load_train_list()
    assert len(train) == 1610
    assert not set(train) & set(ucf.load_test_list())


@needs_annotations
def test_violence_like_subset_has_38_videos():
    subset = ucf.violence_like_test_videos()
    assert len(subset) == 38
    assert "Fighting003_x264" in subset and "Arson011_x264" not in subset


@needs_index
def test_index_covers_all_test_videos_at_30_fps():
    index = ucf.load_index()
    assert len(index) == 290
    assert all(abs(float(r["fps"]) - 30.0) < 0.1 for r in index)
    assert all(int(r["n_frames_decoded"]) > 0 for r in index)


@needs_index
def test_annotation_intervals_start_inside_video():
    frames = {r["video"]: int(r["n_frames_decoded"]) for r in ucf.load_index()}
    for name, video in ucf.load_annotations().items():
        for start, end in video.intervals:
            assert 0 <= start <= end and start < frames[name], name


@needs_index
def test_known_videos_label_counts():
    labels = ucf.test_frame_labels()
    assert labels["Abuse028_x264"].sum() == 240 - 165 + 1
    n_frames_arson011 = 1266  # annotation ends at 1267: second interval is clipped to the decoded video
    assert len(labels["Arson011_x264"]) == n_frames_arson011
    assert labels["Arson011_x264"].sum() == (420 - 150 + 1) + (n_frames_arson011 - 680)
    assert labels["Normal_Videos_003_x264"].sum() == 0
    assert labels["Abuse028_x264"][164] == 0 and labels["Abuse028_x264"][165] == 1


# Rebuild published gt arrays from our labels (evidence in EXP-000-data/log.md).
# Layout: anomalous test videos in Anomaly_Test.txt order, then the 150 normal ones; each video truncated/padded
# to whole 16-frame clips. RTFM: ceil(n/16) clips, end inclusive. VadCLIP: floor(n/16) clips, end exclusive,
# except Shooting032 (+1 clip) and 48 extra zero frames among the normal videos (unidentifiable, all-zero).
GT_RULES = {
    "rtfm": dict(clips=lambda n: -(-n // 16), end_inclusive=True, extra_clips={}),
    "vadclip": dict(clips=lambda n: n // 16, end_inclusive=False, extra_clips={"Shooting032_x264": 1}),
}
RTFM_KNOWN_DIFFERENCES = {"Arson011_x264": 12, "Arson016_x264": 11, "Explosion033_x264": 11, "Shooting015_x264": 12}


def _published_layout(repo):
    rule = GT_RULES[repo]
    annotations = ucf.load_annotations()
    frames = {r["video"]: int(r["n_frames_decoded"]) for r in ucf.load_index()}
    names = [ucf.video_name(r) for r in ucf.load_test_list()]
    names = [n for n in names if annotations[n].is_anomalous] + [n for n in names if not annotations[n].is_anomalous]
    for name in names:
        length = 16 * (rule["clips"](frames[name]) + rule["extra_clips"].get(name, 0))
        yield name, ucf.frame_labels(annotations[name], length, rule["end_inclusive"])


@needs_index
@pytest.mark.parametrize("repo", ["rtfm", "vadclip"])
def test_published_gt_matches_our_labels_per_video(repo):
    if not REFERENCE_GT[repo].exists():
        pytest.skip("reference gt not downloaded")
    published = np.load(REFERENCE_GT[repo]).astype(np.uint8)
    position, differences = 0, {}
    for name, ours in _published_layout(repo):
        segment = published[position : position + len(ours)]
        if (n_diff := int((segment != ours).sum())) or len(segment) != len(ours):
            differences[name] = n_diff
        position += len(ours)
    expected = RTFM_KNOWN_DIFFERENCES if repo == "rtfm" else {}
    assert differences == expected
    assert len(published) - position == (48 if repo == "vadclip" else 0)
    assert not published[position:].any()


@needs_index
def test_total_frames_and_anomalous_frames():
    labels = ucf.test_frame_labels()
    assert sum(len(v) for v in labels.values()) == 1_111_808
    assert sum(int(v.sum()) for v in labels.values()) == 84_331
