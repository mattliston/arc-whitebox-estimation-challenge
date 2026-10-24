# Fixed suffix training pilot: closed

2026-09-05T17:51Z · experiment · rejected · measured

September 5, 2026. Five predeclared epochs completed in 51.97 seconds on the sole MPS job, with 20 train / 4 held-out public networks. Fails the >=8% reduction on every held-out net gate.

Recorded measurement:

    September 5, 2026. Five predeclared epochs completed in 51.97 seconds on the sole MPS job, with 20 train / 4 held-out public networks. The exact prefix through layer 8 remains byte-identical. Shadow adjoints and coupled late-layer finite-difference fidelity had passed before this fit. | Held-out rank | Parent raw | New raw | Ratio | | --- | --- | --- | --- | | 37 | 2.15350826147e-8 | 2.15428219830e-8 | 1.00035938 | | 39 | 1.77858665601e-8 | 1.77753040756e-8 | 0.99940613 | | 42 | 2.31795683708e-8 | 2.32253199215e-8 | 1.00197379 | | 48 | 2.22270149547e-8 | 2.18234341703e-8 | 0.98184278 | Fails the >=8% reduction on every held-out net gate. No 200-net run, epoch/LR sweep, scalar-final refit, port, panel or submission. The shadow predicted these small changes closely; the failure is not a claimed large shadow gain that disappeared on deployment. This closes the fixed warm suffix fit, not all end-to-end algorithms. Evidence: late_transport_training_pilot_v1.json, late_transport_actual_verdict_v1.json and payload_late_transport_pilot_v1.npz. Official bank remains 329926, adjusted 4.98223146018e-9; goal below 1e-9 and the 4e-9/3e-9/2e-9 milestone banks remain unfinished.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/score_late_transport_pilot_v1.py](code/astra_phase2_reexamination_v1/score_late_transport_pilot_v1.py)
- [code/astra_phase2_reexamination_v1/train_late_transport_pilot_v1.py](code/astra_phase2_reexamination_v1/train_late_transport_pilot_v1.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [LATE_TRANSPORT_TRAINING_PILOT_V1_CLOSED.md#L1](LATE_TRANSPORT_TRAINING_PILOT_V1_CLOSED.md#L1)
- [LATE_TRANSPORT_TRAINING_PILOT_V1_CLOSED.md#L3](LATE_TRANSPORT_TRAINING_PILOT_V1_CLOSED.md#L3)
- [LATE_TRANSPORT_TRAINING_PILOT_V1_CLOSED.md#L16](LATE_TRANSPORT_TRAINING_PILOT_V1_CLOSED.md#L16)

<details>
<summary>All result files and records (4)</summary>

- [LATE_TRANSPORT_TRAINING_PILOT_V1_CLOSED.md](LATE_TRANSPORT_TRAINING_PILOT_V1_CLOSED.md) · record
- [late_transport_training_pilot_v1.json](late_transport_training_pilot_v1.json) · result
- [late_transport_actual_verdict_v1.json](late_transport_actual_verdict_v1.json) · result
- [LATE_TRANSPORT_TRAINING_PILOT_V1.md](LATE_TRANSPORT_TRAINING_PILOT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-fixed-suffix-training-pilot-closed) · [Copy provenance](../../../PROVENANCE.json)

