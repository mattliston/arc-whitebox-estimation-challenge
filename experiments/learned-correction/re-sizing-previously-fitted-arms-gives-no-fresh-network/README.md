# Re-sizing previously fitted arms gives no fresh-network signal; two-stage pruning fails its stability rule

2026-10-12T18:26Z · experiment · rejected · measured

radlab's yield analysis: candidate generation 50 (S50)'s 8 strength arms re-sized in candidate generation 51 (S51) sit at chi2 6.1 on 8 df on fresh A41; gains come from first sizings; the read-only two-stage prune changed the frozen move by 26% (rule 10%), so one-stage sizing stayed; candidate generation 52 (S52) was lean and then superseded.

Recorded measurement:

    re-sized arms joint LOO > 0 at every ridge; new arms arrp + diso + dk21n + drsy chi2 19.6 on 4 df (about -0.1% together); supply after S51 0.1-0.3%; S51 freeze LOO -0.186%, check -0.279% (se 0.119, 20/32)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/radlab`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/yield_v1.py](code/yield_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1520](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1520)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1521](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1521)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1522](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1522)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1543](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1543)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1612](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1612)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1615](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1615)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1616](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1616)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1634](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1634)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1746](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1746)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1753](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1753)

<details>
<summary>All result files and records (7)</summary>

- [REPORT_v1_lead_copy.md](../../../records/research/2026-10-12-refit-round-rules-remove-re-sized-arms-second-stage/REPORT_v1_lead_copy.md) · explainer
- [yield_S51_A41p13_v1.json](yield_S51_A41p13_v1.json) · result
- [RETRO_PRUNE_v1.json](RETRO_PRUNE_v1.json) · record
- [FROZEN_POINT_v1.json](../../../records/research/2026-10-12-zero-step-refit-arm-receives-weight-16-54-rounding/FROZEN_POINT_v1.json) · record
- [CHECK_VERDICT_v1.json](CHECK_VERDICT_v1.json) · record
- [SUPERSEDED_v1.json](SUPERSEDED_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Generated refit-after-drop rounds connect sizing, porting, sealed judging and release through verdict files](../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/README.md)
- enables: [Small refit rounds rejected for native insignificance or independent-check failure](../small-refit-rounds-rejected-native-insignificance-or-independent-check/README.md)
- shipped in: [Submission #335618: learned-array, head-row and per-layer strength refit in the lower-request form](../../../submissions/335618/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-re-sizing-previously-fitted-arms-gives-no-fresh-network) · [Copy provenance](../../../PROVENANCE.json)

