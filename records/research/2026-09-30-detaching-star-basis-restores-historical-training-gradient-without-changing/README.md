# Detaching the star basis restores the historical training gradient without changing initial losses

2026-09-30 · finding · informational · measured

The historical training hook computed QR/SVD under no-gradient detached inputs while retaining differentiation through Q.T@A; the80-update reconstruction differentiated all three QR steps. The corrected basis detachment preserved four initial network losses and changed the gradient, without proving why the earlier endpoint failed.

Recorded measurement:

    Initial4 network MSEs exactly equal;first gradient norm7.338744 versus7.237160;all80 old gradient norms were clipped,min1.676/median6.226/max22.692.

## Programs

- [train_segment.py](train_segment.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L329](../2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L329)
- [HILL_CLIMB_CURRENT.md#L320](../2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L320)

<details>
<summary>All result files and records (3)</summary>

- [HILL_CLIMB_CURRENT.md](../2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [trainer_build.json](trainer_build.json) · result
- [initial_forward_check.json](initial_forward_check.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-detaching-star-basis-restores-historical-training-gradient-without-changing) · [Copy provenance](../../../PROVENANCE.json)

