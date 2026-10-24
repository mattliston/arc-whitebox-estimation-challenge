# Product bill map: matrix products use 68% of bill at 0.46–0.51 of dense cost; no useful dense bypass

2026-10-16T20:51:09Z · finding · informational · measured

Two probes on the warm judge network (7103256) decomposed the estimator submitted as #336412 (S2tr) bill: products are 68% of it and the Winograd trees run at 0.46-0.51 of dense (FLOP-optimal); the symmetric covariance sandwich costs about 400 requests per layer for 0.7 GF (13% of requests, 4% of bill); of 258.6 GF, 206 are matmuls dominated by the strip leaf kernel, the input-tree unwind, the Smirnov batch kernel and the d7 padded product, with everything else under 1% each. Non-matmul work (20%) is the Winograd transform adds and subtracts (intrinsic); binding is about 1.5% cold-only. The bill sits at the fast-matmul floor.

Recorded measurement:

    Products 68% of bill at 0.46-0.51 of dense; sandwich ~400 requests/layer for 0.7 GF (13% of requests, 4% of bill); parent 4-strip / 3-strip / two-strip on one 0.21%/1,000 line. Matmuls 206 of 258.6 GF: _hz_leaf 67.4 GF (26%), _ipt_leaf_unwind 55.0 (21%), bb_batch (SM40 Smirnov kernel) 44.1 (17%), _d7_product_kpad 21.7 (8.4%); gop2 message product 1.0 GF; np_update / radial matvecs 0.4.

## Programs

- [probe_products_v1.py](probe_products_v1.py)
- [probe_matmul_v1.py](probe_matmul_v1.py)
- [cand_s2tr.py](cand_s2tr.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4491](../../TEAM_LOG_2026-10-01_to_10-22.md#L4491)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4498](../../TEAM_LOG_2026-10-01_to_10-22.md#L4498)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4508](../../TEAM_LOG_2026-10-01_to_10-22.md#L4508)

<details>
<summary>All result files and records (4)</summary>

- [probe_products_s2tr_7103256.json](probe_products_s2tr_7103256.json) · result
- [probe_matmul_s2tr_7103256.json](probe_matmul_s2tr_7103256.json) · result
- [probe_charges_s2tr_7103256.json](probe_charges_s2tr_7103256.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-product-bill-map-matrix-products-use-68-bill-0) · [Copy provenance](../../../PROVENANCE.json)

