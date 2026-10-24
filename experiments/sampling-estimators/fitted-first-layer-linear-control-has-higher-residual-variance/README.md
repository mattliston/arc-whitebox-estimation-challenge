# Fitted first-layer linear control has higher residual variance than antithetic sampling

2026-09-08 · experiment · rejected · measured

The fixed two-network comparison measured plain, antithetic, fitted-control and optimistic projection-floor variances.

Recorded measurement:

    RESULT_2026_09_08.md:
    | Final-layer TEST statistic | rank35 | rank100 |
    | Plain positive-branch variance |0.06729593059|0.07709202601|
    | Antithetic variance |0.03057988382|0.03489043697|
    | Fitted-control residual variance |0.03313994501|0.03769859738|
    | Optimistic linear-feature projection floor |0.02489093719|0.02840902068|
    | Floor empirical95% lower endpoint |0.02466988635|0.02805269875|
    Numerical checks passed: fixed same-path CPU64/MPS32 controlled-residual

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/first_layer_antithetic_cv_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check.py](code/check.py)

## Evidence

- [RESULT_2026_09_08.md#L3](RESULT_2026_09_08.md#L3)
- [RESULT_2026_09_08.md#L13](RESULT_2026_09_08.md#L13)
- [RESULT_2026_09_08.md#L15](RESULT_2026_09_08.md#L15)
- [RESULT_2026_09_08.md#L16](RESULT_2026_09_08.md#L16)
- [RESULT_2026_09_08.md#L17](RESULT_2026_09_08.md#L17)

<details>
<summary>All result files and records (2)</summary>

- [RESULT_2026_09_08.md](RESULT_2026_09_08.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-fitted-first-layer-linear-control-has-higher-residual-variance) · [Copy provenance](../../../PROVENANCE.json)

