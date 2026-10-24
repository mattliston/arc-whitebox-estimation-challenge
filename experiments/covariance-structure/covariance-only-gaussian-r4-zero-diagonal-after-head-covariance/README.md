# Covariance-only Gaussian R4 (zero-diagonal, after the head) and covariance diagonal precision: +0.22%/+0.26%, closed

2026-10-15T01:44:52Z · experiment · rejected · measured

Two direct covariance corrections at layers 1 to 14 were tried after the feasible head mean: the Gaussian R4 term applied only to the off-diagonal covariance (zero-diagonal, modeled 0.055% bill) and a covariance diagonal precision repair (CDP, modeled 0.023% bill, whose first attempt stopped before layer 0 on a function guard). Both made the two fixed networks worse on average.

Recorded measurement:

    covariance-only R4: paired adjusted +0.215117%, MSE +0.141487%, bill +0.073526% (7100136 net -0.12917%, 7100137 +0.56059%); CDP v2: adjusted +0.255122%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_covariance_only_r4_v1/fragment_covariance_r4_HELD_v2.py](code/astra_covariance_only_r4_v1/fragment_covariance_r4_HELD_v2.py)
- [code/astra_covariance_diagonal_precision_v1/candidate_cdp_HELD_v2.py](code/astra_covariance_diagonal_precision_v1/candidate_cdp_HELD_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3249](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3249)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3251](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3251)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3252](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3252)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3254](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3254)

<details>
<summary>All result files and records (3)</summary>

- [ROOT_CONDITIONAL_GRANT_v1.json](ROOT_CONDITIONAL_GRANT_v1.json) · record
- [DIAGNOSTIC_PROTOCOL_HELD_v2.json](DIAGNOSTIC_PROTOCOL_HELD_v2.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-covariance-only-gaussian-r4-zero-diagonal-after-head-covariance) · [Copy provenance](../../../PROVENANCE.json)

