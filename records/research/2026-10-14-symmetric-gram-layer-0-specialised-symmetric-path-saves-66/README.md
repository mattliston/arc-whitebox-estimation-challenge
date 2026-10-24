# Symmetric Gram at layer 0: specialised symmetric path saves 66.9M cold / 137.9M warm FLOPs (0.026/0.054%), no 0.2% route

2026-10-14T07:52:52Z · finding · rejected · derived

A bounded source and tariff audit of exploiting the symmetry of the layer-0 Gram product found only a small local saving with added requests and changed rounding; naive smaller matmul grids were worse, so no 0.2% path exists.

Recorded measurement:

    66.9M FLOPs cold / 137.9M repeated local charge reduction (~0.026% / 0.054% of a 257G bill)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2808](../../TEAM_LOG_2026-10-01_to_10-22.md#L2808)

<details>
<summary>All result files and records (2)</summary>

- [SOURCE_AND_TARIFF_CONCLUSION_v2.json](SOURCE_AND_TARIFF_CONCLUSION_v2.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-symmetric-gram-layer-0-specialised-symmetric-path-saves-66) · [Copy provenance](../../../PROVENANCE.json)

