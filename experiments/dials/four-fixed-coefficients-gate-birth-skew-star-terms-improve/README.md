# Four fixed coefficients for gate, birth, skew and star terms improve 32-network raw error by 0.295%

2026-10-17T00:21:08Z · experiment · adopted · measured

The four dial means that read negative in the store sweep were stacked as zb4 (GK4 = 1.0, _BFIX_GAMMA = 0.119, GSK = 0.9, SS_BETA = 0.85, all zero bill) and run on the 32 store networks as the only remaining candidate for a valid clean best. It passed its pre-registered gate (mean <= -0.15%, wins >= 20). An audit of all 877 arm results against their controls found only zb4, gk4_10 and gsk_09 qualifying, all covered by zb4; GK4 and GSK act through the live transport gate on inherited third moments. zb4 was carried into the Astra peel stack and the v17 lineage; its public read later was about -0.08% (store -0.3%).

Recorded measurement:

    zb4 32 networks: raw mean -0.295490%, SE .107932%, 21/32 wins, pooled -0.294176%, first bill -0.001902%. Audit (43 arms, 42 with n >= 8; threshold n >= 8 / mean <= -0.15%): zb4 n32 -0.297370% SE .108410 / 21 wins; gk4_10 n23 -0.172929% SE .097866 / 14 wins; gsk_09 n13 -0.154785% SE .147179 / 6 wins; bfix119 faded at n18 to -0.102873%; ott30 raw -0.669331% overcome by +1.011669% first bill, net +0.335518%. ReadonlyBini+W3+zb4 (cf7c0e0a) fixed 8: raw +0.759515% (pooled +0.728262%, 2/8).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_kinkoracle_v1/cand_decomp_zb4_v1.py](code/opus_kinkoracle_v1/cand_decomp_zb4_v1.py)
- [code/opus_kinkoracle_v1/cand_decomp_gk4_10_v1.py](code/opus_kinkoracle_v1/cand_decomp_gk4_10_v1.py)
- [code/opus_kinkoracle_v1/cand_decomp_gsk_09_v1.py](code/opus_kinkoracle_v1/cand_decomp_gsk_09_v1.py)
- [code/astra_crunch_stack_v1/candidate_read_w3_zb4_unpooled_v1.py](code/astra_crunch_stack_v1/candidate_read_w3_zb4_unpooled_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4522](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4522)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4557](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4557)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4567](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4567)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4568](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4568)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4569](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4569)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4570](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4570)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4574](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4574)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4898](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4898)

<details>
<summary>All result files and records (5)</summary>

- [frag_zb4_v1.py.txt](frag_zb4_v1.py.txt) · code
- [AUDIT_v1.json](AUDIT_v1.json) · result
- [PAIRED_ARMS_v1.json](PAIRED_ARMS_v1.json) · result
- [RESULT_MANIFEST_v1.json](RESULT_MANIFEST_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Store dial sweep of 39 zero-bill arms (gate, skew, kurtosis, birth, plane scales, late grows): nothing beyond noise](../store-dial-sweep-39-zero-bill-arms-gate-skew/README.md)

[Research graph](../../../graph/index.html#node=2026-10-17-four-fixed-coefficients-gate-birth-skew-star-terms-improve) · [Copy provenance](../../../PROVENANCE.json)

