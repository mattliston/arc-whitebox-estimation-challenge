# Existing recurrent-encoder direction: no expansion

2026-09-06T04:23Z · experiment · rejected · measured

A single train-only relative-gradient direction used44,144 existing message/encoder parameters; all mean heads, gains, normalizers, moment physics and finalQ stayed frozen. Training ranks35/36/38/40; fixed new-fit holdouts37/39/42/48. Pooled ratio .9910316850;2/4 wins.

Recorded measurement:

    A single train-only relative-gradient direction used44,144 existing message/encoder parameters; all mean heads, gains, normalizers, moment physics and finalQ stayed frozen. Training ranks35/36/38/40; fixed new-fit holdouts37/39/42/48. The scalar least-squares step .02141471511 came only from training tangents and did not reach the .10 relative-group cap. Full-carrier reverse/forward directional derivatives agree to2.05e-7 relative. The independent dynamic NumPy finite-step results are: | Rank | Original raw | New raw | Ratio | | --- | --- | --- | --- | |37|2.12811809538e-8|2.16057574007e-8|1.015252| |39|1.78616289101e-8|1.78167840369e-8|.997489| |42|2.32244332166e-8|2.36819632407e-8|1.019700| |48|2.24393813739e-8|2.09415472550e-8|.933250| Pooled ratio .9910316850;2/4 wins. This fails .90 and3-wins. Do not expand this direction to optimization, a different scalar/group/radius or larger data. This is not a universal limit on all learned encoders. Torch/NumPy actual trajectories differ by at most2.19e-7. Probe118.83s, peakRSS9.37GB, one explicitly authorized nice10 CPU worker,3threads; swap stayed5926.50MiB. CPU finite replay13.84s; NumPy judge24.20s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/judge_encoder_end_to_end_direction_v1.py](code/astra_phase2_reexamination_v1/judge_encoder_end_to_end_direction_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/probe_encoder_end_to_end_direction_v1.py](code/astra_phase2_reexamination_v1/probe_encoder_end_to_end_direction_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/replay_encoder_end_to_end_torch_v1.py](code/astra_phase2_reexamination_v1/replay_encoder_end_to_end_torch_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [ENCODER_END_TO_END_DIRECTION_V1_CLOSED.md#L1](ENCODER_END_TO_END_DIRECTION_V1_CLOSED.md#L1)
- [ENCODER_END_TO_END_DIRECTION_V1_CLOSED.md#L18](ENCODER_END_TO_END_DIRECTION_V1_CLOSED.md#L18)

<details>
<summary>All result files and records (5)</summary>

- [ENCODER_END_TO_END_DIRECTION_V1_CLOSED.md](ENCODER_END_TO_END_DIRECTION_V1_CLOSED.md) · record
- [encoder_end_to_end_direction_v1.json](encoder_end_to_end_direction_v1.json) · result
- [encoder_end_to_end_actual_torch_v1.json](encoder_end_to_end_actual_torch_v1.json) · result
- [encoder_end_to_end_actual_verdict_v1.json](encoder_end_to_end_actual_verdict_v1.json) · result
- [ENCODER_END_TO_END_DIRECTION_DESIGN_V1.md](ENCODER_END_TO_END_DIRECTION_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-existing-recurrent-encoder-direction-no-expansion) · [Copy provenance](../../../PROVENANCE.json)

