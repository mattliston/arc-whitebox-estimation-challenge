# flopscope 0.12 tariffs read from source: uint64 AND/popcount rate 2, zeros 0 F, padding copy charged, trace unwrapped

2026-09-08T10:25Z · finding · informational · measured

Reading the installed flopscope 0.12 sources (data_gate/.venv) established several billing facts used to price proposals: bitwise_and and bitwise_count on uint64 operands are charged at rate 2 each even when popcount returns uint8; zeros/empty creation costs 0 F while the copyto of a padded input slice is charged per element; outer(dm,dm) with the same operand is billed with the symmetric n(n+1)/2 base count; and trace returns an unwrapped NumPy scalar, so a subsequent scalar power is not metered. These are faithful prices of the current source, not permission to move arithmetic outside the meter.

Recorded measurement:

    uint64 bitwise_and 2 F, bitwise_count 2 F, bounded 32-bit word reduction 1 F per element; zeros((mp,832)) 0 F then copyto 1024*832 = 851968 F; self-outer n(n+1)/2 base; trace 2n, scalar **2 0 F (not 32 F)

## Evidence

- [BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L27](../2026-09-08-fixed-4x4-bit-popcount-products-cost-2-34-2/BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L27)
- [BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L35](../2026-09-08-fixed-4x4-bit-popcount-products-cost-2-34-2/BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md#L35)
- [HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L26](../2026-09-08-two-unimplemented-exact-micro-cuts-denominator-reuse-dead-terminal/HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L26)
- [HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L40](../2026-09-08-two-unimplemented-exact-micro-cuts-denominator-reuse-dead-terminal/HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L40)
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L207](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L207)

<details>
<summary>All result files and records (3)</summary>

- [BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md](../2026-09-08-fixed-4x4-bit-popcount-products-cost-2-34-2/BITSLICED_POPCOUNT_PRICE_PREFLIGHT_2026_09_08_V1.md) · record
- [HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md](../2026-09-08-two-unimplemented-exact-micro-cuts-denominator-reuse-dead-terminal/HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md) · record
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-flopscope-0-12-tariffs-read-source-uint64-popcount-rate) · [Copy provenance](../../../PROVENANCE.json)

