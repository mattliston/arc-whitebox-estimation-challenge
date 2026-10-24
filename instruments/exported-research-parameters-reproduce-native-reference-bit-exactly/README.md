# Exported research parameters reproduce the native reference bit-exactly

2026-09-22T21:47Z · tool · informational · measured

export_params.py writes a refit's .pt parameters into the five literal statements of a native candidate; re-exporting the original parameters into cand_drop16 reproduces the parent's native prediction and bill exactly, validating the deploy path from torch refits to native files.

Recorded measurement:

    cand_drop16_reexport.py r35: util 0.18829, mse 1.648017257382719e-08 (bit-exact to the parent)

## Programs

- [code/export_params.py](code/export_params.py)
- [code/score_variant.py](code/score_variant.py)
- [code/paired_eval.py](code/paired_eval.py)

## Evidence

- [NOTES.md#L88](../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L88)

<details>
<summary>All result files and records (3)</summary>

- [score_reexport_r35.log](score_reexport_r35.log) · log
- [paired_sanity.log](paired_sanity.log) · log
- [NOTES.md](../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-22-exported-research-parameters-reproduce-native-reference-bit-exactly) · [Copy provenance](../../PROVENANCE.json)

