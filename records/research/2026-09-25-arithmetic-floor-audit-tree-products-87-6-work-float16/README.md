# Arithmetic floor audit: tree products are 87.6% of work; float16 receives no discount

2026-09-25T00:20Z · finding · informational · measured

Product depths were already near their leaf-arithmetic floors. Measured dtype rates ruled out half precision and complex packing as substantial work-saving mechanisms.

Recorded measurement:

    2026-09-24 17:20 [engineer2] BILL BY DEPTH on S16 (altbasis/depth_bill_census.py, r100, 358.5 G): tree products 87.6%; depth 6 43.8% at 0.500 of dense 2mkn (leaf floor 0.449): colpack 22.1%, batched cov/star 16.6%, singles 5.1%; depth 5 42.9% at 0.564 (floor 0.513): reads (_packed_read5) 25.0%, other 15.6% (covariance halves 3.7%), W5 2.3%; depth <=4 0.9%. Measured dtype rates: float16 bills 1.0 like float32; complex64 matmul 4.0 (even complex x real). Strategic memo to the lead: kernel is at its practical floor (no >=5% kernel lever; fast-matmul alternatives <=1-1.5%; remaining depth -0.6% for +4.4 s); the -30% is VOLUME (reads + pushes ~59% of the bill; Luna ~half our variable volume). Levers: dataflow association/sharing audit, read packing at depth 6 (-3..-4%), fewer old-cohort columns ranked by MSE per Gflop; crazy: Khatri-Rao collapse of long-k reads.

## Programs

- [depth_bill_census.py](depth_bill_census.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L481](../../TEAM_LOG_2026-09-22_to_09-26.md#L481)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-arithmetic-floor-audit-tree-products-87-6-work-float16) · [Copy provenance](../../../PROVENANCE.json)

