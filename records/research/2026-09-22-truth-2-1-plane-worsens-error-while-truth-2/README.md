# Truth (2,1) plane worsens error while truth (2,2), third, and fourth cumulants improve it

2026-09-22T21:04Z · finding · informational · measured

Feeding 8M-sample Monte Carlo truths into the pair maps at layers 12-15 shows the learned closure is tuned to the card's own (2,1) plane (the true K21 plane makes the final error 2.4x worse) while true per-unit third and fourth cumulants and the true (2,2) plane help modestly; the headroom from truth cumulant supplies before any refit is about 7-12%.

Recorded measurement:

    truth K21 -> 4.03e-8 (2.4x worse); truth K22 (incl. per-unit k4) -> 1.507e-8 (-8.6%); both -> 3.77e-8; true k3 diag -> 1.596e-8 (-3.2%); true per-unit k4 -> 1.536e-8 (-6.8%); both -> 1.456e-8 (-11.7%); control 1.648e-8 (r35)

## Evidence

- [NOTES.md#L45](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L45)
- [NOTES.md#L47](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L47)

<details>
<summary>All result files and records (3)</summary>

- [oracle_planes_331824_r35.log](oracle_planes_331824_r35.log) · log
- [oracle_diag_331824_r35.log](oracle_diag_331824_r35.log) · log
- [NOTES.md](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-truth-2-1-plane-worsens-error-while-truth-2) · [Copy provenance](../../../PROVENANCE.json)

