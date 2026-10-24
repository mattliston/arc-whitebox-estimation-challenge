# Cubic Edgeworth channel groups exceeded the available FLOP budget

2026-08-25 · finding · rejected · derived

Adjudicated at design stage by exact synthetic flopscope billing, with no source build and no data access. The L5 group is not implementable (the host carries no fifth-order cumulant state). Every implementable lambda-cubed Edgeworth channel group exceeds the exact 2^41 headroom at the current host total: third-cumulant Edgeworth operator (L3)*fourth-cumulant Edgeworth operator (L4) needs 69,547,720,704 added FLOPs over fourteen layers against a 66,547,897,095 headroom including the J correction, L3^3 needs 86,934,650,880, and both together 156,482,371,584 - before counting weight-construction cost.

Recorded measurement:

    Adjudicated at design stage by exact synthetic flopscope billing, with no source build and no data access. The L5 group is not implementable (the host carries no fifth-order cumulant state). Every implementable lambda-cubed Edgeworth channel group exceeds the exact 2^41 headroom at the current host total: L3*L4 needs 69,547,720,704 added FLOPs over fourteen layers against a 66,547,897,095 headroom including the J correction, L3^3 needs 86,934,650,880, and both together 156,482,371,584 - before counting weight-construction cost. Together with the T closure, the series-refinement family is measured out: within the budget the Q24 closure's expansion orders sit at their cost frontier and the sub-leading terms that fit do not matter.

## Evidence

- [SYNTHETIC_CHANNEL_BILLING.json#L1](SYNTHETIC_CHANNEL_BILLING.json#L1)
- [U_DESIGN_CLOSURE.json#L1](U_DESIGN_CLOSURE.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [SYNTHETIC_CHANNEL_BILLING.json](SYNTHETIC_CHANNEL_BILLING.json) · result
- [U_DESIGN_CLOSURE.json](U_DESIGN_CLOSURE.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-cubic-edgeworth-channel-groups-exceeded-available-flop-budget) · [Copy provenance](../../../PROVENANCE.json)

