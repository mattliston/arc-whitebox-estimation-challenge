# Strided Winograd arena views make transform additions 3.7-5.5x slower than contiguous arrays

2026-09-13 · experiment · rejected · measured

A microbenchmark isolated maximum-shape arena striding as the cause of slow transform additions. Exact-shape reshaping was charged one FLOP per element, whereas slicing, unstack, zeros and empty allocation were free.

Code availability: No separate saved microbenchmark program was found. RECEIPTS.md line 29 records the 3.7-5.5x timing comparison; the later exact-shape variants are implementations motivated by it, not the missing benchmark program.

Recorded measurement:

    Strided views3.7-5.5x slower; transform additions65% of native time; reshape1 flop/element.

The measured result is retained here. A separate executable program was not identified; this entry is indexed as a result record rather than a code directory.

## Evidence

- [RECEIPTS.md#L28](../2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L28)

<details>
<summary>All result files and records (1)</summary>

- [RECEIPTS.md](../2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-strided-winograd-arena-views-make-transform-additions-3-7) · [Copy provenance](../../../PROVENANCE.json)

