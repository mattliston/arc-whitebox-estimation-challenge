# Increasing fourth-cumulant rank from 2048 to 8192 did not repair the missing planes

2026-09-04 · finding · informational · measured

fourth-cumulant canonical-polyadic carrier (CP4) RANK DOES NOT HELP (r35, CP4_CAP 2048 / 4096 / 8192; k4dump_cp4cap*): fourth-order pre-plane ratio vs truth at L8 fourth-cumulant two-pair plane (K22) 0.58 / 0.56 / 0.56, fourth-cumulant triple-single plane (K31) 0.44 / 0.43 / 0.43, k4 0.58 / 0.56 / 0.56 (corr unchanged 0.98 / 0.70); host 8.62 / 9.05 / 9.22e-8 (worse) at util 0.600 / 0.666 / 0.762. The card's fourth-order shortfall (planes and k4 at 0.4-0.6 of truth with the RIGHT direction) is not carrier capacity: it is how the card ASSEMBLES/EMITS its fourth-order content each layer (same conclusion as T-45 from the other side). Hypothesis (T-85 test): `_reset_stars` fits pairing_alpha = <K22 target, v v^T + 2 C∘C>/<..> by least squares and subtracts 3 alpha v v from k4 and alpha-scaled planes - a projection that also removes the part of the CONNECTED content aligned with the disconnected plane (the truth K22 is collective/rank-8), producing exactly a direction-preserving under-scale of K22 and k4 with the same layer profile.

Recorded measurement:

    - T-84 CP4 RANK DOES NOT HELP (r35, CP4_CAP 2048 / 4096 / 8192; k4dump_cp4cap*): fourth-order pre-plane ratio vs truth at L8 K22 0.58 / 0.56 / 0.56, K31 0.44 / 0.43 / 0.43, k4 0.58 / 0.56 / 0.56 (corr unchanged 0.98 / 0.70); host 8.62 / 9.05 / 9.22e-8 (worse) at util 0.600 / 0.666 / 0.762. The card's fourth-order shortfall (planes and k4 at 0.4-0.6 of truth with the RIGHT direction) is not carrier capacity: it is how the card ASSEMBLES/EMITS its fourth-order content each layer (same conclusion as T-45 from the other side). Hypothesis (T-85 test): `_reset_stars` fits pairing_alpha = <K22 target, v v^T + 2 C∘C>/<..> by least squares and subtracts 3 alpha v v from k4 and alpha-scaled planes - a projection that also removes the part of the CONNECTED content aligned with the disconnected plane (the truth K22 is collective/rank-8), producing exactly a direction-preserving under-scale of K22 and k4 with the same layer profile.

## Evidence

- [LANE_g3_tracker.md#L726](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L726)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-increasing-fourth-cumulant-rank-2048-8192-did-not-repair) · [Copy provenance](../../../PROVENANCE.json)

