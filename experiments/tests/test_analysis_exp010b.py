"""EXP-010b: the seed/output-root switch of exp010 and the degeneracy rule of S2."""

from pathlib import Path

from edgelab.analysis import exp010, exp010b


def test_configure_changes_seed_and_output_root_and_can_be_restored():
    old = (exp010.SEED, exp010.OUT, exp010.ROWS, exp010.MLFLOW_EXPERIMENT, exp010.EXPERIMENT_DIR)
    try:
        assert exp010.SEED == 1  # EXP-010 default is unchanged
        exp010b.configure()
        assert exp010.SEED == 2 and exp010.OUT.name == "exp010b" and exp010.ROWS == exp010.OUT / "rows.jsonl"
        assert exp010.MLFLOW_EXPERIMENT == "EXP-010b"
    finally:
        exp010.configure(old[0], old[1], old[3], old[4])
    assert (exp010.SEED, exp010.OUT, exp010.MLFLOW_EXPERIMENT) == (1, old[1], "EXP-010")


def test_a_cell_is_degenerate_when_either_arm_has_fewer_than_200_windows():
    assert exp010b.is_degenerate({"num_segments": 34}, {"num_segments": 5000})
    assert exp010b.is_degenerate({"num_segments": 5000}, {"num_segments": 30})
    assert not exp010b.is_degenerate({"num_segments": 200}, {"num_segments": 200})
