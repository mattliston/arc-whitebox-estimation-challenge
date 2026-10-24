# Late-layer variance correction required low-noise structured supply

2026-09-04 · experiment · informational · measured

NOISE TOLERANCE OF THE LATE-LAYER v CHANNEL (operator ON, half v at layers 9-15): supply from the 32M truth r35 2.153e-8 (-20%) / r100 1.961e-8 (-29%); from the 8M truth (per-unit v noise ~0.05% vs dv 0.1-0.3%, i.e. ~3-10% noise energy at these layers) r35 2.208e-8 (-18%) / r100 2.122e-8 (-24%). A supply with per-unit corr ~0.97 keeps ~85% of the value; the channel needs the exact per-unit field's DIRECTION (T-100c) but tolerates a few percent of incoherent noise energy. Graph reconciled to V45 (reconcile_20260903_late_layer_v45.py, validated); memory p2-late-born-error-and-lowrank-residual written.

Code availability: The shared retrofit instrument retains truth-forcing and layer-window controls. A dedicated launch script for the exact 8M/32M truth comparison was not identified.

Recorded measurement:

    - T-101 NOISE TOLERANCE OF THE LATE-LAYER v CHANNEL (operator ON, half v at layers 9-15): supply from the 32M truth r35 2.153e-8 (-20%) / r100 1.961e-8 (-29%); from the 8M truth (per-unit v noise ~0.05% vs dv 0.1-0.3%, i.e. ~3-10% noise energy at these layers) r35 2.208e-8 (-18%) / r100 2.122e-8 (-24%). A supply with per-unit corr ~0.97 keeps ~85% of the value; the channel needs the exact per-unit field's DIRECTION (T-100c) but tolerates a few percent of incoherent noise energy. Graph reconciled to V45 (reconcile_20260903_late_layer_v45.py, validated); memory p2-late-born-error-and-lowrank-residual written.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L781](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L781)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-late-layer-variance-correction-required-low-noise-structured-supply) · [Copy provenance](../../../PROVENANCE.json)

