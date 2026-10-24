# Thirty per-unit features cannot predict late pre-variance errors across networks

2026-10-20T08:31:31Z · experiment · rejected · measured

A 30-feature ridge model achieved in-sample R^2 <= 0.18 but cross-network R^2 near zero or negative. This closed the tested per-unit producer for the exact-variance oracle gain.

Recorded measurement:

      (2) dv_i is NOT predictable from per-unit card features (v, mu, mu^2, k3, k4, gate, alpha, W-pushed phi/Phi/ones/s phi/post mean, diagonal transports, K21 row sums, products with v): 30-feature ridge, in-sample R^2 <= 0.18, OUT-of-sample R^2 ~ 0 or negative on the other net; best single |corr| 0.17 (mu^2). The per-unit variance lever (-18/-26% oracle) has no per-unit producer.
    [fable-lead] 2026-10-20T08:31:31Z FOUR OFFLINE CLOSURES on the late-layer error (nets 35/100, card quantities from the k21dump files at L8-14 vs the 32M mcall truth; lead/runs/fable_oct19_kc_v1/{dcov_structure,dv_regress,k22cov_offline,mehler_offline,edge2_offline}_v1.py):

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/dv_regress_v1.py](code/dv_regress_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6149](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6149)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6147](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6147)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-thirty-per-unit-features-cannot-predict-late-pre-variance) · [Copy provenance](../../../PROVENANCE.json)

