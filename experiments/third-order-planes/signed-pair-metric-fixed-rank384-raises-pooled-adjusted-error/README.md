# Signed-pair metric at fixed rank384 raises pooled adjusted error to 1.010772448328348

2026-09-19 · experiment · rejected · measured

Only the basis-entry metric changed, separating this test from the earlier combined rank-and-metric reduction.

Recorded measurement:

    # Fixed rank384 signed-pair metric: development failure
    Rank384, age4, trained heads, physical factor coordinates, Lazy schedule,
    previously absent same-rank comparison; the original PAIR256 joint
    | Development rank | Lazy raw MSE | Pair384 raw MSE | Raw ratio | Own-cold adjusted ratio |
    | 35 | 1.870602849007794e-08 | 1.867859647500963e-08 | 0.998533520085 | 0.999268168499 |
    | 100 | 1.842453417527493e-08 | 1.88229765452842e-08 | 1.02162564145 | 1.02237056117 |
    Pooled adjusted ratio: **1.010772448328348**.
    Rank35 improves slightly; rank100 worsens. Both raw and adjusted
    improvement on both networks and pooled adjusted<=.99 were required.
    These are two reused ordinary1B development networks, not an
    Rank35 complete cold FLOPs: 398080917697 -> 398373796713 (+292879016, 0.073573%).
    Rank100 complete cold FLOPs: 400893063472 -> 401185375192 (+292311720, 0.072915%).
    action with maximum relative error1.5600663146273522e-7. Signed,
    Full outputs are finite/nonnegative; first5 parent layers, own repeat,
    warm/cold100 output bytes and cache retirement pass. Changed later
    Four full calls total 1566584423454 billed prediction FLOPs.
    Peak RSS including self-reports: 6501187584 bytes. Zero added swap.
    All four supervised stages exit0 and retire their process groups.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/pair384_metric_v1/build.py](code/codex_resume_20260919_v1/pair384_metric_v1/build.py)
- [code/codex_resume_20260919_v1/pair384_metric_v1/candidate.py](code/codex_resume_20260919_v1/pair384_metric_v1/candidate.py)
- [code/codex_resume_20260919_v1/pair384_metric_v1/finalize.py](code/codex_resume_20260919_v1/pair384_metric_v1/finalize.py)
- [code/codex_resume_20260919_v1/pair384_metric_v1/metric_fnp.py](code/codex_resume_20260919_v1/pair384_metric_v1/metric_fnp.py)
- [code/codex_resume_20260919_v1/pair384_metric_v1/score_dev.py](code/codex_resume_20260919_v1/pair384_metric_v1/score_dev.py)
- [code/codex_resume_20260919_v1/pair384_metric_v1/unit.py](code/codex_resume_20260919_v1/pair384_metric_v1/unit.py)
- [code/astra_phase2_reexamination_v1/current_comp_shared_gain_slice_v1/build_v1/estimator.py](code/astra_phase2_reexamination_v1/current_comp_shared_gain_slice_v1/build_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/current_comp_shared_old384_v1/fnp.py](code/astra_phase2_reexamination_v1/current_comp_shared_old384_v1/fnp.py)
- [code/astra_phase2_reexamination_v1/current_comp_shared_pair256_v1/compose.py](code/astra_phase2_reexamination_v1/current_comp_shared_pair256_v1/compose.py)
- [code/astra_phase2_reexamination_v1/current_comp_shared_pair256_v1/metric_fnp.py](code/astra_phase2_reexamination_v1/current_comp_shared_pair256_v1/metric_fnp.py)
- [code/codex_resume_20260919_v1/lazy_rotation_objective_v1/candidate.py](code/codex_resume_20260919_v1/lazy_rotation_objective_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L1](NEGATIVE_RESULT.md#L1)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L6](NEGATIVE_RESULT.md#L6)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)
- [REEXAMINATION_INDEX_20260919.md#L1364](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1364)
- [REEXAMINATION_INDEX_20260919.md#L1367](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1367)

<details>
<summary>All result files and records (5)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-signed-pair-metric-fixed-rank384-raises-pooled-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)

