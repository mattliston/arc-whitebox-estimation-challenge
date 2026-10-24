# Mean-sensitivity selection of 384 weight columns raises pooled adjusted error to 1.0282819350797

2026-09-20 · experiment · rejected · measured

A local Gaussian ReLU sensitivity to the third cumulant selected weight columns for the shared-basis range sketch.

Recorded measurement:

    # Fixed consumer-directed Omega384 closes negative
    On exact submitted Lazy, selecting the384 current-weight columns by modeled
    local Gaussian ReLU mean sensitivity to kappa3 worsened both development
    | 35 | 1.01498896423881 | 1.01499282836795 | 398082433217 |
    | 100 | 1.04167549458359 | 1.0416794324917 | 400894578992 |
    Pooled adjusted ratio **1.0282819350797**, versus the
    fixed<=.99 requirement. Both individual raw/adjusted-improvement gates also
    fail. Own-cold bills include initialization and each adds1,515,520 FLOPs.
    The ordinary1B development references are reused, not fresh held-out data.
    8.81867176786427e-07. First5 parent output rows are
    byte-exact; own repeated35 and warm/cold100 whole output files match exactly.
    1565420104062 billed FLOPs. Peak observed/self-reported
    RSS was6393610240 bytes; added swap0. All numeric caches
    it does not close all rank384 bases or full-state endpoint-risk compression.
    Official Lazy#331599 stays at adjusted3.8164492657795035e-9. The4e-9 bank is

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/consumer_omega384_v1/build.py](code/consumer_omega384_v1/build.py)
- [code/consumer_omega384_v1/candidate.py](code/consumer_omega384_v1/candidate.py)
- [code/consumer_omega384_v1/consumer_fnp.py](code/consumer_omega384_v1/consumer_fnp.py)
- [code/consumer_omega384_v1/score_dev.py](code/consumer_omega384_v1/score_dev.py)
- [code/consumer_omega384_v1/unit.py](code/consumer_omega384_v1/unit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L1](NEGATIVE_RESULT.md#L1)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L11](NEGATIVE_RESULT.md#L11)

<details>
<summary>All result files and records (4)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-mean-sensitivity-selection-384-weight-columns-raises-pooled-adjusted) · [Copy provenance](../../../PROVENANCE.json)

