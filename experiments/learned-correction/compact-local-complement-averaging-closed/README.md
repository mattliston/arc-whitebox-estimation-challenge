# Compact local complement averaging closed

2026-09-05T15:50Z · experiment · rejected · measured

24 positive teacher trajectories match parent means/features/sigma/head outputs byte-for-byte. Scalar mean complement is certified to1.78e-15. Held-out37/39/42/48, layers10..15,1e9 post labels: parent4.80933777e-9; averaged positive/negative frozen heads9.64980053e-9, ratio2.00647. 0/4 held-out nets improved.

Recorded measurement:

    24 positive teacher trajectories match parent means/features/sigma/head outputs byte-for-byte. Scalar mean complement is certified to1.78e-15. But the fixed negative feature branch, which deliberately reused the positive compact K21 mask, is not a full active-mask-reselected rerun. Held-out37/39/42/48, layers10..15,1e9 post labels: parent4.80933777e-9; averaged positive/negative frozen heads9.64980053e-9, ratio2.00647. 0/4 held-out nets improved. This fails the5% information gate. No200-row fitting, recursive port, damping or submission. The exact ReLU identity is not refuted; the reused compact feature branch and frozen head were not a useful implementation of it. Receipt: relu_complement_probe_verdict_v1.json.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/generate_relu_complement_probe_v1.py](code/astra_phase2_reexamination_v1/generate_relu_complement_probe_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/run_relu_complement_probe_v1.py](code/astra_phase2_reexamination_v1/run_relu_complement_probe_v1.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [RELU_COMPLEMENT_PROBE_V1_CLOSED.md#L1](RELU_COMPLEMENT_PROBE_V1_CLOSED.md#L1)

<details>
<summary>All result files and records (3)</summary>

- [RELU_COMPLEMENT_PROBE_V1_CLOSED.md](RELU_COMPLEMENT_PROBE_V1_CLOSED.md) · record
- [relu_complement_probe_verdict_v1.json](relu_complement_probe_verdict_v1.json) · result
- [RELU_COMPLEMENT_PROBE_V1.md](RELU_COMPLEMENT_PROBE_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-compact-local-complement-averaging-closed) · [Copy provenance](../../../PROVENANCE.json)

