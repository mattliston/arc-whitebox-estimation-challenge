# Truth marginal inputs reduce error 12.7% at the final layer and 30.1% across layers 8–15

2026-09-22T21:13Z · finding · informational · measured

Replacing the per-unit variance, third and fourth cumulants by 8M truths inside the pair maps and the final nonlinear scalar map prices the per-unit input error: the last layer alone is worth 12.7% and layers 8-15 30.1%, while truth variance at early layers hurts (co-adaptation). A separate test shows the marginal formula with exact 8M inputs reproduces the truth within the inputs' own noise at every layer, so the inputs, not the formula, are the bottleneck.

Recorded measurement:

    L15 only: true v -3.8% | true k3+k4 -8.0% | v+k3+k4 -12.7%; L12-15: v -10.5% | v34 -24.0%; L8-15: 34 -24.6% | v -22.1% | v34 -30.1%; L1-15: 34 -26.3% | v34 -13.9%; final formula with exact inputs: L15 8.1e-9 vs noise floor 8.4e-9 (Gaussian-only 1.9e-7)

## Programs

- [oracle_units.py](oracle_units.py)
- [final_formula_test.py](final_formula_test.py)

## Evidence

- [NOTES.md#L67](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L67)
- [NOTES.md#L70](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L70)

<details>
<summary>All result files and records (3)</summary>

- [oracle_units_r35.log](oracle_units_r35.log) · log
- [final_formula_test_r35.json](final_formula_test_r35.json) · result
- [NOTES.md](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-truth-marginal-inputs-reduce-error-12-7-final-layer) · [Copy provenance](../../../PROVENANCE.json)

