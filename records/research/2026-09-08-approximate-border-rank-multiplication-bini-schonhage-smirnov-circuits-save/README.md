# Approximate (border-rank) multiplication: Bini, Schonhage, Smirnov circuits save at most 15% of products; no prototype

2026-09-08T07:55Z · finding · rejected · derived

Real approximate circuits outside the exact catalog (Bini <2,2,3> rank 10, Schonhage <3,3,3> rank 21, Smirnov rank 20) replace one Winograd level with ideal product-only improvements of 4.7619%, 11.1111% and 15.3439%; on the archived 1024x832x8192 product Bini plus three or four Winograd levels gives explicit subtotals below the metered four-level (D4) cost, but Bini's O(epsilon) truncation amplifies rounding as O(u/epsilon), FP64 leaves alone would cost more than the whole present FP32 product, and no cancellation-controlled circuit with a complete ledger exists. The route was closed at its precision prerequisite.

Recorded measurement:

    ideal one-level improvements 4.7619% / 11.1111% / 15.3439%; archived D4 cost 8,417,538,304 F vs Bini+3 levels 8,157,917,896 F and +4 levels 7,589,323,384 F (padded rows 1032/1056); FP64 leaves alone 15,465,623,040 F; installed FNP 0.12 zeros creation 0 F, padding copy 1024*832 = 851968 F

## Evidence

- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L184](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L184)
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L196](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L196)
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L213](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L213)
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L221](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L221)

<details>
<summary>All result files and records (2)</summary>

- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md) · record
- [CONTINUATION_2026_09_08_0105_PDT.md](CONTINUATION_2026_09_08_0105_PDT.md) · record

</details>

## Related work

- ← supersedes: [Bini <3,2,2> approximate-product defect is dense: correcting four edge terms costs 3.24 GF against a 0.26 GF margin](../2026-09-08-bini-3-2-2-approximate-product-defect-dense-correcting/README.md)

[Research graph](../../../graph/index.html#node=2026-09-08-approximate-border-rank-multiplication-bini-schonhage-smirnov-circuits-save) · [Copy provenance](../../../PROVENANCE.json)

