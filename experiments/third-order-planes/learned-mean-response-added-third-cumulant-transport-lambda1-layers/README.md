# Learned mean response added to the third-cumulant transport (lambda1, layers 1-14): bill +0.31%, MSE -0.007%, rejected

2026-10-14T05:22:10Z · experiment · rejected · measured

Adding a learned mean response term to the third-cumulant (plane) transport at layers 1 to 14 cost 0.31% of bill and changed the final MSE by -0.007%, so the fresh screen rejected it before any out-of-sample run.

Recorded measurement:

    native MAX bill log +0.003124577129930257 (0.3125%); fresh 32 (16 pairs): mse_log -0.007251083171116768%, bill_log 0.3124577129930257%, net_log 0.30520662982190894% (SE 0.0051108265855919705%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_learned_mean_transport_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check_derivatives_synthetic_v2.py](code/check_derivatives_synthetic_v2.py)
- [code/check_derivatives_synthetic_v1.py](code/check_derivatives_synthetic_v1.py)
- [code/conditional_tangent_math_UNVALIDATED_v1.py](code/conditional_tangent_math_UNVALIDATED_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2641](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2641)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2664](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2664)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2700](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2700)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-learned-mean-response-added-third-cumulant-transport-lambda1-layers) · [Copy provenance](../../../PROVENANCE.json)

