# Ascending retained-index ordering fails all four covariance-head response gates

2026-09-20 · finding · rejected · measured

Sorting retained indices preserved the selected sets on identical inputs but did not resolve the nonlinear finite-difference response.

Recorded measurement:

    Exact official331684 remains the adjusted3.5844144212297072e-9 bank,100/100 completed.
    All existing thresholds, retained sets on identical inputs, FP32 kernels and head coefficients remain fixed.
    The original16-head response study remains closed negative; the same epsilon1/16 and coordinates0/8
    Four original rank36/head0 replays match all16 old output bytes and complete bills.
    All canonical perturbations preserve their own base first8 rows; repeat output/bill/trace is exact.
    |36|0|0.704147584|0.769302048|1.26512434|1.77738321|False|
    |36|8|0.576542657|0.630717398|0.822498317|0.997001807|False|
    |38|0|0.933951542|0.984229815|1.3594502|1.4050212|False|
    |38|8|0.457045779|0.516723013|0.847319223|0.948807684|False|
    Each cell requires spacing SNR>=20 and both response ratios<=.10.
    Across all six rank38 probes, selected indices and masks equal the canonical base: True.
    Input scores do change. Both rank38 response cells still fail, so differing selector sets/orders
    All23 guarded groups retired;19 full native calls;zero added swap;
    peak sampled group RSS6274433024 bytes; no new samples,datasets or GPU.

## Programs

- [build.py](build.py)
- [candidate.py](candidate.py)
- [capture.py](capture.py)
- [common.py](common.py)
- [finalize.py](finalize.py)
- [qualify.py](qualify.py)
- [run.py](run.py)
- [unit.py](unit.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L9](RESULT.md#L9)
- [REEXAMINATION_INDEX_20260919.md#L790](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L790)
- [REEXAMINATION_INDEX_20260919.md#L793](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L793)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-ascending-retained-index-ordering-fails-all-four-covariance-head) · [Copy provenance](../../../PROVENANCE.json)

