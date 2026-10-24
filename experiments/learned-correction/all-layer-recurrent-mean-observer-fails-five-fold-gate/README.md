# All-layer recurrent mean observer fails five-fold gate: adjusted ratio 1.00422797606219

2026-09-20 · experiment · rejected · measured

A 102-coefficient observer was fitted on frozen current trajectories and all eight late-layer targets with a declared provisional cost allowance.

Recorded measurement:

    Official parent #331684 remains adjusted 3.5844144212297072e-9, with all 100 networks completed.
    This separate 102-coefficient FP64 mean observer used the current frozen parent trajectory,
    existing 1B mean targets at all eight late layers, and five fixed network folds.
    All 20 training captures reproduce the parent output bytes and complete bills.
    - Pooled adjusted recurrent/parent ratio: 1.00422797606219 (required <= 0.95).
    - Recurrent/local-control ratio: 1.00962106184144 (required <= 0.99).
    - Parent wins: 7/20 networks and 2/5 folds (required >= 12 and >= 4).
    - Worst network ratio: 1.03382065847319 (required <= 1.20).
    Scores use the declared provisional 1GF increment; no native added-cost qualification is claimed.
    Resources: 32 full native calls; 36 retired guarded groups; zero added swap;
    peak sampled group RSS 6425755648 bytes;
    300950479 artifact bytes before closeout; no new samples, downloads, or GPU.
    The approximately 1e-9 goal remains active and unachieved.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/full_lower_multilayer_mean_observer_v1/capture.py](code/full_lower_multilayer_mean_observer_v1/capture.py)
- [code/full_lower_multilayer_mean_observer_v1/check.py](code/full_lower_multilayer_mean_observer_v1/check.py)
- [code/full_lower_multilayer_mean_observer_v1/common.py](code/full_lower_multilayer_mean_observer_v1/common.py)
- [code/full_lower_multilayer_mean_observer_v1/design.py](code/full_lower_multilayer_mean_observer_v1/design.py)
- [code/full_lower_multilayer_mean_observer_v1/finalize.py](code/full_lower_multilayer_mean_observer_v1/finalize.py)
- [code/full_lower_multilayer_mean_observer_v1/fit.py](code/full_lower_multilayer_mean_observer_v1/fit.py)
- [code/full_lower_multilayer_mean_observer_v1/fit_v2.py](code/full_lower_multilayer_mean_observer_v1/fit_v2.py)
- [code/full_lower_multilayer_mean_observer_v1/observer_math.py](code/full_lower_multilayer_mean_observer_v1/observer_math.py)
- [code/full_lower_multilayer_mean_observer_v1/observer_math_v2.py](code/full_lower_multilayer_mean_observer_v1/observer_math_v2.py)
- [code/full_lower_multilayer_mean_observer_v1/prepare_capture.py](code/full_lower_multilayer_mean_observer_v1/prepare_capture.py)
- [code/full_lower_multilayer_mean_observer_v1/prepare_fit.py](code/full_lower_multilayer_mean_observer_v1/prepare_fit.py)
- [code/full_lower_multilayer_mean_observer_v1/prepare_fit_v2.py](code/full_lower_multilayer_mean_observer_v1/prepare_fit_v2.py)

<details>
<summary>Additional programs and dependencies (8)</summary>

- [code/full_lower_multilayer_mean_observer_v1/qualify_capture.py](code/full_lower_multilayer_mean_observer_v1/qualify_capture.py)
- [code/full_lower_multilayer_mean_observer_v1/run_analysis.py](code/full_lower_multilayer_mean_observer_v1/run_analysis.py)
- [code/full_lower_multilayer_mean_observer_v1/run_capture.py](code/full_lower_multilayer_mean_observer_v1/run_capture.py)
- [code/full_lower_multilayer_mean_observer_v1/unit.py](code/full_lower_multilayer_mean_observer_v1/unit.py)
- [code/full_lower_multilayer_mean_observer_v1/unit_v2.py](code/full_lower_multilayer_mean_observer_v1/unit_v2.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L6](RESULT.md#L6)
- [REEXAMINATION_INDEX_20260919.md#L823](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L823)
- [REEXAMINATION_INDEX_20260919.md#L826](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L826)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-all-layer-recurrent-mean-observer-fails-five-fold-gate) · [Copy provenance](../../../PROVENANCE.json)

