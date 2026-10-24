# Product-dataflow audit finds at most 0.5% work saving from reassociation or shared operands

2026-09-25T00:28Z · finding · informational · measured

Tracing operands and product uses showed that the existing associations were already optimal and the proposed shared-operand reads were not packable.

Recorded measurement:

    2026-09-24 17:28 [engineer2] LEVER 1 (dataflow association / sharing / Khatri-Rao) FALSIFIED: <= -0.5% of the bill. Identity-tracking census of S16 r100 (altbasis/dataflow_census.py + dataflow_analyze.py; 608 products = 95.2% of the bill, every operand traced through views, every product output's uses counted): 28 single-use product->product chains, all already at the cheapest association (cov w^T(Cw) ties); long-k read operands are (V o V)[live rows] and V[rows] o lambda over concatenated cohort columns (no pair structure, no Khatri-Rao collapse); apparent shared roots are disjoint old/young column slices (17% of the bill, not packable), the old384 bank reused by 6 successive layers (cross-layer tree cache only, -0.13% for ~300 MB), and one W^T slice per layer shared by the colpack newborn push and the W5 port (fold: ~-0.3%). Lever 2 (colpack for reads) dies too: reads share no operand within a layer. Bill by caller: old384 products 24.1%, young-square reads 21.9%, newborn bank push 21.5%, cov/star 20.1%, W5 3.4%; ~69% of the bill is in products whose k or n is the cohort column count (4.6k-22.6k).
    - 2026-09-24 17:50 lead: ENGINEER2 DATAFLOW AUDIT FALSIFIED (<= -0.5%). Census: 608 products, 95.2% of the bill, every operand traced. Association 0% (all 28 chains already optimal), Khatri-Rao none, shared operands not packable (reads share nothing within a layer, so colpack-for-reads is dead). BILL BY CALLER: old384 24.1%, young-square reads 21.9%, newborn bank push 21.5%, cov/star block 20.1%, W5 port 3.4%. ~69% sits in products whose contraction or width is the cohort column count. So the only >= 5% bill lever is fewer columns or a cheaper old-tier representation, which puts the weight on covhunter's read-aware frame (old384 = 24%). Small items: W5 into the push tree -0.3%.

## Programs

- [dataflow_census.py](dataflow_census.py)
- [dataflow_analyze.py](dataflow_analyze.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L513](../../TEAM_LOG_2026-09-22_to_09-26.md#L513)
- [TEAM_LOG_2026-09-22_to_09-26.md#L514](../../TEAM_LOG_2026-09-22_to_09-26.md#L514)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-product-dataflow-audit-finds-most-0-5-work-saving) · [Copy provenance](../../../PROVENANCE.json)

