# First-network view creation raises residual time and magnifies the operand-load setup cost

2026-10-02T12:15Z · finding · informational · derived

On the platform the first network after a setup carries 1,495 requests that build views of tree keys used for the first time in the process and about 320 that build once-per-process constants, at 14.6 microseconds each (twice the average request); its residual is 1.04-1.10 times the others' and in S37, S38 and S39 the run's largest residual is such a network (S39: 0.313-0.347 s against at most 0.322 s for the other 93). 19% of public runs have a worker set up again mid-run. Building those views in the setup's slack would take up to 1,500 requests off each first network; all 266 deferred fusions save only about 280 requests per network.

Recorded measurement:

    first vs other networks mean residual: S35 0.316 / 0.304 (x1.04), S36 0.307 / 0.282 (x1.09), S37 0.318 / 0.303 (x1.05), S38 0.322 / 0.293 (x1.10), S39 0.327 / 0.300 (x1.09); first network 43,565 requests vs 41,682 warm; first-use views 1,495 (_hz_leaf_views 352, _l3_quad 399, _l3o_pairs 276, _rs_basis_views 178, _ipt_side 151, _lvf_rs_level1 81, _ipt_view 58), constants about 320; +0.0274 s at 14.6 microseconds each; S39 + 200 requests: first networks 0.348 s largest without the lever, about 0.338 s (views of prebound keys) or 0.322 s (all view families and constants) with it

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L293](../../TEAM_LOG_2026-10-01_to_10-22.md#L293)
- [TEAM_LOG_2026-10-01_to_10-22.md#L303](../../TEAM_LOG_2026-10-01_to_10-22.md#L303)
- [TEAM_LOG_2026-10-01_to_10-22.md#L307](../../TEAM_LOG_2026-10-01_to_10-22.md#L307)
- [TEAM_LOG_2026-10-01_to_10-22.md#L315](../../TEAM_LOG_2026-10-01_to_10-22.md#L315)
- [TEAM_LOG_2026-10-01_to_10-22.md#L317](../../TEAM_LOG_2026-10-01_to_10-22.md#L317)
- [RESULT.md#L1037](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md#L1037)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-first-network-view-creation-raises-residual-time-magnifies-operand) · [Copy provenance](../../../PROVENANCE.json)

