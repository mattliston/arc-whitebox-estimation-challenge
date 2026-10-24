# Query input completeness audit

2026-09-05T22:18Z · finding · informational · measured

The v1 shape-only q4/s4 price was incomplete. The current bank only fills the active-active CP/star preplane, plus every scalar diagonal. The v2 ledger charges full CP repeated-plane reads (including appended source columns), and the extra compact-star bilinear/cold reads.

Recorded measurement:

    The v1 shape-only q4/s4 price was incomplete. The current bank only fills the active-active CP/star preplane, plus every scalar diagonal. The certified full-width R@U query and its repeated-index restoration require the FULL input K21; masked entries cannot silently stand in for those. The v2 ledger charges full CP repeated-plane reads (including appended source columns), and the extra compact-star bilinear/cold reads. The v1 receipts are preserved, not treated as a passed complete cost gate. No width1024 accuracy or fitting work was run on that assumption.

## Evidence

- [THIN_FOURTH_INPUT_PLANE_AUDIT_V2.md#L1](THIN_FOURTH_INPUT_PLANE_AUDIT_V2.md#L1)

<details>
<summary>All result files and records (1)</summary>

- [THIN_FOURTH_INPUT_PLANE_AUDIT_V2.md](THIN_FOURTH_INPUT_PLANE_AUDIT_V2.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-query-input-completeness-audit) · [Copy provenance](../../../PROVENANCE.json)

