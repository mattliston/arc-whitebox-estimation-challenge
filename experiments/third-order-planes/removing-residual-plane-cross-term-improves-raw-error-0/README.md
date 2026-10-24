# Removing the residual-plane cross term improves raw error 0.438% and adjusted error 2.911%

2026-09-25T05:17Z · experiment · adopted · measured

Deleting the previously rank-32 star cross term removed its range finder and thin products while improving held-out raw error. The remaining star term was retained.

Recorded measurement:

    - 2026-09-24 22:17 lead: s1 = 0 BEATS rank-32 s1 (engineer2; 4-net in-process vs S21s1): bill -2.48% (its range finder and thin products go too), raw -0.49% (-0.51/-1.25/+0.29/-0.54), adjusted -2.96%; vs S21 -4.90%; ~-240 requests/net.
    2026-09-24 22:17 [engineer2] s1 = 0 TEST on S21s1 (#332236; lead item 1): the star's s1 term dropped (sub = s2/3), in-process parity vs S21s1 on 35/100/37/39: bill -2.479% (per net -2.38..-2.55%; the rank-32 s1's range finder and thin products go too), MSE -0.489% (per net -0.51/-1.25/+0.29/-0.54), adjusted -2.956% (vs S21: bill -4.43%, MSE -0.50%, adjusted -4.90%); ~-240 requests/net. Better than free on 4 nets (pooled-raw SE ~0.35%). Candidate lrroute/S21s1_s1zero_mem.py a49da660 = S21s1 30aea66e + s1zero_fragment.py; validate/einsum PASS. VALID40 vs the S21s1 rows (paired_valid40_e2_S21s1lr.json) queued behind the heavy gate (yields to S22); proposed bar adjusted <= -1.5% CI excl 0, raw <= +0.5%. Physicist told to hold the rank-32 s1 hook lines pending the verdict (if s1=0 wins: hook sub = s2/3).
    - 2026-09-24 23:00 lead: s1 = 0 VALID40 PASS (engineer2 S21s1_s1zero_mem.py a49da660 vs the S21s1 rows): adjusted -2.911% [CI -3.107, -2.725], 40/40 (per-net -4.50..-1.72%); raw -0.438% [CI -0.641, -0.243], 28/40. A real raw gain plus the bill cut: s1 was net-harmful. S23 (with s1=0 + Nystrom join + loopR1) VALID40 started 22:59; smoke util .1412 (S21s1 .1476), requests 31,966 (-2,051/net). Expected ~-5.9% vs S21s1 -> ~2.38e-9.
    2026-09-24 23:42 [engineer2] s1=0 VALID40 PASS (lrroute/S21s1_s1zero_mem.py a49da660 vs the S21s1 rows paired_valid40_e2_S21s1lr.json; porter harness): adjusted -2.911% [-3.107, -2.725] 40/40 (per net -4.50..-1.72, median -2.82); raw -0.438% [-0.641, -0.243] 28/40 (significant improvement); util -2.482% [-2.508, -2.458]. Bar (lead): adj <= -1.5% CI excl 0, raw <= +0.5% -> PASS. Chained with S21s1: ~-4.6% adjusted vs S21. VALID40_DONE marker + lrroute/valid40_e2_S21s1z_bar.json written 22:56 for fire_chain_S23.sh (S23 contains s1=0); no standalone s1=0 residual gate (lead); residual-study probe queued after S23's chain.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer2_v1/lrroute/S21s1_s1zero_mem.py](code/fable_engineer2_v1/lrroute/S21s1_s1zero_mem.py)
- [code/fable_engineer2_v1/lrroute/s1zero_fragment.py](code/fable_engineer2_v1/lrroute/s1zero_fragment.py)
- [code/fable_porter_v1/fire_chain_S23.sh](code/fable_porter_v1/fire_chain_S23.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L700](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L700)
- [TEAM_LOG_2026-09-22_to_09-26.md#L704](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L704)
- [TEAM_LOG_2026-09-22_to_09-26.md#L739](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L739)
- [TEAM_LOG_2026-09-22_to_09-26.md#L763](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L763)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [paired_valid40_e2_S21s1lr.json](paired_valid40_e2_S21s1lr.json) · result
- [valid40_e2_S21s1z_bar.json](valid40_e2_S21s1z_bar.json) · result

</details>

## Related work

- shipped in: [Submission #332246: cross-term removal and Nyström old-cohort joins](../../../submissions/332246/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-removing-residual-plane-cross-term-improves-raw-error-0) · [Copy provenance](../../../PROVENANCE.json)

