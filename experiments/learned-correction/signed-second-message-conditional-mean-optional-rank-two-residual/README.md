# Signed second-message conditional mean with optional rank-two residual: native parity delivery fails

2026-10-14T03:21:07Z · experiment · rejected · measured

A signed M2 conditional mean for the learned correction (mean-only, or with a frozen rank-2 centered residual) was priced natively, passed the 48-network fresh screen for both variants, but failed the formal native delivery (matched-C parity) for both, so its out-of-sample pool was cancelled unread.

Recorded measurement:

    native price worst-seed log bill: MEAN -0.181080468%, RANK2 -0.156861786%; fresh 48: MEAN mse_log -0.072662047 / bill_log -0.181080468 / net_log -0.253742515 (SE 0.09512111)%, RANK2 mse_log +0.04493909 / bill_log -0.156861786 / net_log -0.111922696 (SE 0.046757935)%; native parity 7 calls: formal delivery FAIL for both; OOS cancelled at 26 of 48 results

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_signed_m2_mean_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_source_v1.py](code/build_source_v1.py)
- [code/baseline_e2_messages_v1.py](code/baseline_e2_messages_v1.py)
- [code/basis_literal_v1.py](code/basis_literal_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2477](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2477)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2508](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2508)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2534](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2534)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2538](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2538)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2558](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2558)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2573](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2573)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2583](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2583)

<details>
<summary>All result files and records (4)</summary>

- [SOURCE_READY_v1.json](SOURCE_READY_v1.json) · record
- [ASTRA_OWNED_RETIREMENT_v1.json](ASTRA_OWNED_RETIREMENT_v1.json) · record
- [ASTRA_OWNED_RETIREMENT_v1-ceb73eff.json](ASTRA_OWNED_RETIREMENT_v1-ceb73eff.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-signed-second-message-conditional-mean-optional-rank-two-residual) · [Copy provenance](../../../PROVENANCE.json)

