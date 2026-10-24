# Block-major Winograd layout would save about 2.5k requests but add about 0.2% bill; rewrite rejected

2026-09-13 · experiment · rejected · measured

A source-level layout analysis replaced four quadrant indexing calls with three unstack operations. Scratch aliasing and input/output conversion cost made the proposed rewrite unattractive.

Code availability: Explicitly evaluated on paper and rejected, per RECEIPTS.md line 57. No implemented block-major prototype is claimed.

Recorded measurement:

    Approximately-2.5k requests; +(mk+kn+mn) per product; approximately+0.2% bill.

The measured result is retained here. A separate executable program was not identified; this entry is indexed as a result record rather than a code directory.

## Evidence

- [RECEIPTS.md#L57](../2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L57)

<details>
<summary>All result files and records (1)</summary>

- [RECEIPTS.md](../2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-block-major-winograd-layout-would-save-about-2-5k) · [Copy provenance](../../../PROVENANCE.json)

