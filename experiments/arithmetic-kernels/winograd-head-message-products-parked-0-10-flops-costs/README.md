# Winograd head-message products parked: -0.10% FLOPs costs 420 operations at rounding-level error

2026-10-12T19:44:54Z · experiment · rejected · measured

The six GNN message products of the head were computed as one Strassen-Winograd level on the (6, n, n) power stack (A-side sums once per layer, 7 half products per batch, one einsum for the 4 output quadrants): metered FLOPs -0.1013% / -0.0969% on 7100136 / 7100137 for +422 / +420 charged operations, with a final-layer prediction change rms 3.1e-6 (MSE +0.03% / -0.44%, a rounding draw); the pre-registered native check stopped at 4 of 8 networks (score -0.075 / -0.542 / +0.763 / -0.548%, MSE mean -0.001%, metered FLOPs -0.0998% every network), v2 read -0.0952% metered FLOPs for +258 requests. Because its larger-error kernel puts the expected net at only -0.035..-0.095%, which an 8-network seal cannot resolve, it was kept out of the stacks and parked for a final polish.

Recorded measurement:

    v1 bill -0.1013% / -0.0969%, ops +422 / +420, change rms 3.1e-6, MSE +0.03% / -0.44%; check 4 of 8: score -0.075 / -0.542 / +0.763 / -0.548% (mean log -0.100%, se 0.309), MSE mean -0.001% (sd ~0.6% per network), bill -0.0998%; v2 bill -0.0952%, requests +258, rms 3.05e-6, final MSE +0.38%; expected net -0.035..-0.095%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/scout_native`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_gnnw_v1.py](code/fragment_gnnw_v1.py)
- [code/sn3g_s50tw/candidate.py](code/sn3g_s50tw/candidate.py)
- [code/sn3_s50tw/candidate.py](code/sn3_s50tw/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1573](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1573)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1650](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1650)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1653](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1653)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1654](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1654)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1657](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1657)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1734](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1734)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1766](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1766)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-winograd-head-message-products-parked-0-10-flops-costs) · [Copy provenance](../../../PROVENANCE.json)

