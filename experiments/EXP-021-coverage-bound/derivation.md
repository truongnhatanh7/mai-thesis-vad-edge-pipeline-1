# EXP-021 derivation: how high can frame-level AUC go when some frames have no scored person?

## Setting

Frame-level AUC pools every test frame of every video. Let there be `nA` anomalous and `nN` normal frames.
Some frames contain no person that the pose pipeline scores ("uncovered"). The scorer gives every uncovered frame
the **most-normal score**: the lowest anomaly score that any covered frame receives (STG-NF's rule, `variant_scores`
in `pose/score_dir.py`). Call this value the *floor*, so all uncovered frames are **tied** at the floor.

Coverage: `cA` = share of anomalous frames that are covered; `cN` = share of normal frames that are covered.
(These are exactly the `coverage_anomalous` / `coverage_normal` logged by `score_dir.py`.)

AUC = probability that a random anomalous frame scores higher than a random normal frame, a tie counting 1/2
(the sklearn / Mann-Whitney convention):

    AUC = (1 / (nA nN)) * sum over pairs (a, n) of  [ 1 if s_a > s_n,  1/2 if s_a = s_n,  0 otherwise ].

## The four blocks of pairs

Split the `nA nN` pairs by whether each member is covered. The shares are:

| block | share of all pairs | what the pair looks like |
|---|---|---|
| covered anomalous vs covered normal | cA cN | scored by the model |
| covered anomalous vs uncovered normal | cA (1 - cN) | normal frame sits at the floor |
| uncovered anomalous vs covered normal | (1 - cA) cN | anomalous frame sits at the floor |
| uncovered anomalous vs uncovered normal | (1 - cA)(1 - cN) | both at the floor: always a tie |

Best case for the model ("covered frames scored perfectly"): every covered anomalous frame outscores every normal frame.
Then the first two blocks contribute their full share (each pair is a win, value 1 at most).

The third block is where coverage costs us. The uncovered anomalous frame has the floor score, and every covered
normal frame has a score `>=` floor (the floor is the minimum over covered frames). So the pair is a win for the normal
frame or a tie, never a win for the anomalous one: value `0` if the covered normal frame is strictly above the floor
(the generic case with continuous scores; only the single frame that defines the floor can equal it), value at most `1/2`
if it sits exactly on the floor.

The fourth block is all ties: value `1/2` (tie-as-half) or `0` (pessimistic).

## Bounds

Adding the blocks (continuous scores, covered normal frames strictly above the floor):

    tie-as-half:   AUC <= cA cN + cA (1 - cN) + 0 + (1/2)(1 - cA)(1 - cN)
                        = cA + (1 - cA)(1 - cN) / 2                           ("bound_half")

    pessimistic (ties count as losses):   AUC <= cA cN + cA (1 - cN) = cA     ("bound_pess")

Reading: an uncovered anomalous frame can never be ranked above a normal frame that has a model score, so the best AUC is
the share of anomalous frames that are covered, plus half of the pairs where both frames are at the floor. Note `cN`
only enters through the tie block: more normal coverage *lowers* the half-credit from uncovered-vs-uncovered pairs, but
the bound is dominated by `cA`.

If covered normal frames are allowed to sit exactly on the floor (all ties credited 1/2 in the third block as well), the
largest value any scoring can reach is

    AUC <= cA + (1 - cA)/2 = (1 + cA)/2                                       ("bound_loose")

which does not depend on `cN` and is a bound for **any** scoring of the covered frames, not only continuous ones.
Ordering: `bound_pess <= bound_half <= bound_loose`. Sanity checks: full coverage (cA = cN = 1) gives 1 for all three;
no coverage (cA = 0) gives `(1 - cN)/2` (half) -> 1/2 when cN = 0 (everything tied, AUC 0.5), and 0 (pess).
`bound_pess` is a bound only for the tie-as-loss AUC; a tie-as-half AUC can exceed it (it does in the data), which is why
`bound_half` is the primary bound.

## What the bound does and does not assume

1. Uncovered frames carry exactly the floor score. This is true of the *raw* per-frame scores. The pipeline then
   applies a **temporal smoother** (offline Gaussian for `paper`, causal half-Gaussian for `live`) over the whole video,
   which spreads covered scores onto neighbouring uncovered frames. Those frames are then no longer at the floor, and
   the premise fails. The bound should therefore be near-tight only when coverage is high and uncovered stretches are
   long, and may be slightly violated at low coverage. `results.md` measures how often (and also gives the
   post-hoc bound on *effective* coverage, counted on the final smoothed scores, where the premise holds exactly).
2. Covered frames scored perfectly is a best case, so the bound is an upper envelope, not a prediction. The slack
   `bound - AUC` is the part of the AUC gap due to scoring quality of covered frames, not coverage.
3. Coverage is pooled over frames, matching the pooled-frame AUC. (Per-video AUC averaging would give a different bound.)
