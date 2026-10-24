# Normalizing incoming weight columns to squared norm2 fails: pooled adjusted ratio 1.002748104451459

2026-09-20 · experiment · rejected · measured

A positive-diagonal network reparameterization preserved exact-network real-arithmetic behavior but did not improve the approximate estimator.

Recorded measurement:

    squared norm2, with final scale1. FP64 scale arithmetic and FP32 stored
    | 35 | 1.870602849007794e-08 | 1.865189844818023e-08 | 0.997106278229 | 0.997806269429 |
    | 100 | 1.842453417527493e-08 | 1.855402639447829e-08 | 1.00702824929 | 1.00773024685 |
    Pooled adjusted ratio **1.002748104451459**.
    The gate requires raw AND adjusted wins on BOTH networks and pooled<=.99.
    These are two reused ordinary1B development networks, not independent
    Rank35 cold bill 398080917697->398360379521 (+0.070202%). First-layer maximum absolute difference 9.53674316406e-07 (limit2e-5).
    Rank100 cold bill 400893063472->401172525296 (+0.069710%). First-layer maximum absolute difference 8.34465026855e-07 (limit2e-5).
    Maximum scaled direct-network output error 5.031942980904821e-08.
    Four full calls pass own repeat, warm/cold100 bytes, restored first-layer
    tolerance, finite nonnegative16x1024 output and original cache retirement.
    Total full-call charge 1566531889278F; peak RSS 6610763776B.
    All four guarded stages exit0/retire with zero added swap. No numerical
    Official bank stays Lazy #331599 adjusted 3.816449265779503e-09.
    The4e-9 milestone is achieved;3e-9,2e-9,~1e-9 remain active/unmet.
    Standing user authorization covers future qualified Phase2 uploads.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/unit_norm_gauge_v1/build.py](code/unit_norm_gauge_v1/build.py)
- [code/unit_norm_gauge_v1/candidate.py](code/unit_norm_gauge_v1/candidate.py)
- [code/unit_norm_gauge_v1/finalize.py](code/unit_norm_gauge_v1/finalize.py)
- [code/unit_norm_gauge_v1/gauge_fnp.py](code/unit_norm_gauge_v1/gauge_fnp.py)
- [code/unit_norm_gauge_v1/score_dev.py](code/unit_norm_gauge_v1/score_dev.py)
- [code/unit_norm_gauge_v1/unit.py](code/unit_norm_gauge_v1/unit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L16](NEGATIVE_RESULT.md#L16)
- [NEGATIVE_RESULT.md#L17](NEGATIVE_RESULT.md#L17)
- [NEGATIVE_RESULT.md#L19](NEGATIVE_RESULT.md#L19)
- [REEXAMINATION_INDEX_20260919.md#L1078](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1078)
- [REEXAMINATION_INDEX_20260919.md#L1085](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1085)

<details>
<summary>All result files and records (5)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-normalizing-incoming-weight-columns-squared-norm2-fails-pooled-adjusted) · [Copy provenance](../../../PROVENANCE.json)

