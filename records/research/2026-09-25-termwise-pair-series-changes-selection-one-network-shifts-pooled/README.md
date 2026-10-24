# Termwise pair-series changes selection on one network and shifts pooled validation 0.084%

2026-09-25T05:48Z · finding · informational · measured

A small floating-point change crossed a selection boundary on one validation network. Subsequent gains were required to use the same implementation for both comparison arms.

Recorded measurement:

    - 2026-09-24 22:48 [physicist] Engineer's E-67 caveat noted: the pair series moves VALID16 +0.084% pooled at j05 ep2 on the jfitR1 card, almost all from net 42 (+1.18%, a selection knife edge). loopR2's bar is same-card (its own valid_before on loop3), so it is unaffected; loopR2 vs loopR1 comparisons cross cards (pair series, s1) and are indicative only. loopR2 still waits for a CPU slot (4 heavy). E-68 MPS refit card is certified and awaits the lead's GO.
      E-67 addendum: the pair-series drop-in moves VALID16 +0.084% pooled (net 42 +1.18%, a knife-edge selection); same-card comparisons are unaffected, but valid_before numbers shift across the switch.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L724](../../TEAM_LOG_2026-09-22_to_09-26.md#L724)
- [TEAM_LOG_2026-09-22_to_09-26.md#L727](../../TEAM_LOG_2026-09-22_to_09-26.md#L727)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-termwise-pair-series-changes-selection-one-network-shifts-pooled) · [Copy provenance](../../../PROVENANCE.json)

