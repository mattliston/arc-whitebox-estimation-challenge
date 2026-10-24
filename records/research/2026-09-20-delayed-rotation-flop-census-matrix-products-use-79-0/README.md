# Delayed-rotation FLOP census: matrix products use 79.0/80.2% of cost; newborn propagation 21.2%, old-tier reads 18.8%

2026-09-20 · finding · informational · measured

A research wrapper around the original charge method reproduced the exact cold bills of the accepted lazy-rotation card (#331599) on nets 35/100 and reconciled every charge by caller, source line, layer and operation shape. Removing all remaining coefficient-rotation charges would save only 2.124%/2.107% and the GNN step at most about 1.12%, whereas reaching 3e-9 at unchanged raw error needs a 21.392902% uniform cost cut; the adjusted floor at the 0.1 multiplier is 2.169264529783277e-9, so accuracy work is necessary for 2e-9 or 1e-9.

Recorded measurement:

    complete bills 398080917697 F (net 35) / 400893063472 F (net 100); caller shares (35/100): _push_bank_shared_newborn 21.209%/21.994%, _shared_old384_read 18.813%/18.594%, _slr_reconstruct 13.543%/13.418%, _shared_old384_push 11.007%/10.937%, _star_compact 10.327%/10.241%, _slr_inner 8.425%/8.239%, _cov_compact 7.679%/7.633%, _slr_join_factors 2.124%/2.107%, gnn_step 1.125%/1.117%; native matmul 79.007%/80.180%; outermost _mm 92.701%/92.759% over 335 calls; cold reshape 5.156%/4.893%; recent pushes plus shared-plane caller 40.022%/40.588%; old reconstruction/inner/rotation 24.091%/23.764%

## Evidence

- [COST_REPORT.md#L3](COST_REPORT.md#L3)
- [COST_REPORT.md#L13](COST_REPORT.md#L13)
- [COST_REPORT.md#L14](COST_REPORT.md#L14)
- [COST_REPORT.md#L15](COST_REPORT.md#L15)
- [COST_REPORT.md#L30](COST_REPORT.md#L30)
- [COST_REPORT.md#L45](COST_REPORT.md#L45)
- [COST_REPORT.md#L48](COST_REPORT.md#L48)
- [COST_REPORT.md#L82](COST_REPORT.md#L82)

<details>
<summary>All result files and records (4)</summary>

- [COST_REPORT.md](COST_REPORT.md) · result
- [result.json](result.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-delayed-rotation-flop-census-matrix-products-use-79-0) · [Copy provenance](../../../PROVENANCE.json)

