# Late rank-8 covariance congruence correction: fixed pilot rejected

2026-09-05T18:44Z · experiment · rejected · measured

The target-free action-response certificate passed on35/100, including byte-identical full-card baseline and both central half-step tests. A fixed five-epoch20-train/4-held-out warm pilot then trained the current full-card suffix mean head and a new rank8 positive-semidefinite (PSD)-preserving covariance action together. All four prefixes through layer9 remain byte-identical.

Recorded measurement:

    The target-free action-response certificate passed on35/100, including byte-identical full-card baseline and both central half-step tests. A fixed five-epoch20-train/4-held-out warm pilot then trained the current full-card suffix mean head and a new rank8 PSD-preserving covariance action together. Training was one MPS job,46.95seconds. No larger fit or sweep was run. | Held-out rank | Parent raw final MSE | New raw final MSE | Ratio | | --- | ---: | ---: | ---: | | 37 | 2.15350826147e-8 | 2.17984803517e-8 | 1.01223110 | | 39 | 1.77858665601e-8 | 1.75264555900e-8 | .98541477 | | 42 | 2.31795683708e-8 | 2.31580061698e-8 | .99906978 | | 48 | 2.22270149547e-8 | 2.20795623625e-8 | .99336606 | All four prefixes through layer9 remain byte-identical. The shadow predictions were similarly flat; this is not an apparent large training gain disappearing in deployment. It fails the preregistered8%-on-each-net gate. Do not run more epochs, learning rates, action scales, ranks, nets, 200-net fits, candidate ports, panels or submissions in this pilot family.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_late_congruence_fidelity_v1.py](code/astra_phase2_reexamination_v1/check_late_congruence_fidelity_v1.py)
- [code/astra_phase2_reexamination_v1/current_congruence_shadow_v1.py](code/astra_phase2_reexamination_v1/current_congruence_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/late_congruence_training_shadow_v1.py](code/astra_phase2_reexamination_v1/late_congruence_training_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/astra_phase2_reexamination_v1/radial_late_congruence_probe_v1.py](code/astra_phase2_reexamination_v1/radial_late_congruence_probe_v1.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/score_late_congruence_pilot_v1.py](code/astra_phase2_reexamination_v1/score_late_congruence_pilot_v1.py)
- [code/astra_phase2_reexamination_v1/train_late_congruence_pilot_v1.py](code/astra_phase2_reexamination_v1/train_late_congruence_pilot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [LATE_CONGRUENCE_PILOT_V1_CLOSED.md#L1](LATE_CONGRUENCE_PILOT_V1_CLOSED.md#L1)
- [LATE_CONGRUENCE_PILOT_V1_CLOSED.md#L9](LATE_CONGRUENCE_PILOT_V1_CLOSED.md#L9)
- [LATE_CONGRUENCE_PILOT_V1_CLOSED.md#L15](LATE_CONGRUENCE_PILOT_V1_CLOSED.md#L15)

<details>
<summary>All result files and records (4)</summary>

- [LATE_CONGRUENCE_PILOT_V1_CLOSED.md](LATE_CONGRUENCE_PILOT_V1_CLOSED.md) · record
- [late_congruence_fidelity_v1.json](late_congruence_fidelity_v1.json) · result
- [late_congruence_training_pilot_v1.json](late_congruence_training_pilot_v1.json) · result
- [late_congruence_actual_verdict_v1.json](late_congruence_actual_verdict_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-late-rank-8-covariance-congruence-correction-fixed-pilot-rejected) · [Copy provenance](../../../PROVENANCE.json)

