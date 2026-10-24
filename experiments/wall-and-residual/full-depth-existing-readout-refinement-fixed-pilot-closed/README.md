# Full-depth existing-readout refinement: fixed pilot closed

2026-09-05T20:37Z · experiment · rejected · measured

The five-epoch,910-coefficient pilot completed in1121.94seconds on one nice10 CPU worker with3threads. Maximum process RSS10.61GB; swap stayed 5966.50MiB. This is a negative optimization result with a faithful derivative, not the earlier frozen-K3 transport failure.

Recorded measurement:

    The five-epoch,910-coefficient pilot completed in1121.94seconds on one nice10 CPU worker with3threads. Maximum process RSS10.61GB; swap stayed 5966.50MiB. It used the newly certified complete CP/star/trace derivative, dynamic selections, and the frozen scalar-final bank head. The independent NumPy evaluation agrees with the torch final MSE to at most0.0059%. | New-fit holdout | Parent raw MSE | Final pilot raw MSE | Ratio | | --- | ---: | ---: | ---: | |35-series rank37|2.12811809538e-8|2.15163014792e-8|1.01104828| |rank39|1.78616289101e-8|1.79584053087e-8|1.00541812| |rank42|2.32244332166e-8|2.33509181514e-8|1.00544620| |rank48|2.24393813739e-8|2.21423815482e-8|0.98676435| Only1/4 improves and the mean does not improve. This fails the declared 15% mean gain plus improvement on every holdout. No radius, epoch, model capacity or expanded-dataset sweep for this fixed-readout refinement. The20% relative trust radius is active only at layer1; that does not justify a radius sweep. No intermediate checkpoint is selected after the fact. Payload SHA256:708d92593f299d708c6be53ae1220ff2a459fe9e699ca536a8d9936c2dfcb43e. Receipts:full_carrier_readout_pilot_v1.json and full_carrier_readout_actual_r37/r39/r42/r48_v1.json. All five epoch artifacts are preserved. Usable official bank329926 remains4.98223146018e-9 adjusted.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/score_full_carrier_readout_pilot_v1.py](code/astra_phase2_reexamination_v1/score_full_carrier_readout_pilot_v1.py)
- [code/astra_phase2_reexamination_v1/train_full_carrier_readout_pilot_v1.py](code/astra_phase2_reexamination_v1/train_full_carrier_readout_pilot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [FULL_CARRIER_READOUT_PILOT_V1_CLOSED.md#L1](FULL_CARRIER_READOUT_PILOT_V1_CLOSED.md#L1)
- [FULL_CARRIER_READOUT_PILOT_V1_CLOSED.md#L7](FULL_CARRIER_READOUT_PILOT_V1_CLOSED.md#L7)
- [FULL_CARRIER_READOUT_PILOT_V1_CLOSED.md#L19](FULL_CARRIER_READOUT_PILOT_V1_CLOSED.md#L19)

<details>
<summary>All result files and records (3)</summary>

- [FULL_CARRIER_READOUT_PILOT_V1_CLOSED.md](FULL_CARRIER_READOUT_PILOT_V1_CLOSED.md) · record
- [full_carrier_readout_pilot_v1.json](full_carrier_readout_pilot_v1.json) · result
- [FULL_CARRIER_READOUT_PILOT_V1.md](FULL_CARRIER_READOUT_PILOT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-full-depth-existing-readout-refinement-fixed-pilot-closed) · [Copy provenance](../../../PROVENANCE.json)

