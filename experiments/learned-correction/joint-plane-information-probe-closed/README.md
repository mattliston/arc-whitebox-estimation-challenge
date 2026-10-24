# Joint-plane information probe: closed

2026-09-05T07:01Z · experiment · rejected · measured

All24 new trajectories preserve the frozen full-pair host, sigma, and all30 old features byte-for-byte. No current-layer correction enters the added 48 response-weighted joint-plane features.

Recorded measurement:

    All24 new trajectories preserve the frozen full-pair host, sigma, and all30 old features byte-for-byte. No current-layer correction enters the added 48 response-weighted joint-plane features. Twenty train nets, four fixed held-out nets37/39/42/48, layers10..15, one ridge scale, no sweep. Held-out residual MSE against stored1e9 means after the frozen teacher-forced parent head: base4.80933777312e-9; old-feature control4.82406653499e-9; augmented joint-plane probe4.80789159607e-9. New/control ratio.9966470, only0.34% gain, below the preregistered5% information gate.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/fit_joint_plane_probe_v1.py](code/astra_phase2_reexamination_v1/fit_joint_plane_probe_v1.py)
- [code/astra_phase2_reexamination_v1/generate_joint_plane_probe_v1.py](code/astra_phase2_reexamination_v1/generate_joint_plane_probe_v1.py)
- [code/astra_phase2_reexamination_v1/joint_plane_features_v1.py](code/astra_phase2_reexamination_v1/joint_plane_features_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/run_joint_plane_probe_rows_v1.py](code/astra_phase2_reexamination_v1/run_joint_plane_probe_rows_v1.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [JOINT_PLANE_PROBE_V1_CLOSED.md#L1](JOINT_PLANE_PROBE_V1_CLOSED.md#L1)

<details>
<summary>All result files and records (4)</summary>

- [JOINT_PLANE_PROBE_V1_CLOSED.md](JOINT_PLANE_PROBE_V1_CLOSED.md) · record
- [joint_plane_probe_rows_v1.json](joint_plane_probe_rows_v1.json) · result
- [joint_plane_probe_verdict_v1.json](joint_plane_probe_verdict_v1.json) · result
- [JOINT_PLANE_PROBE_V1.md](JOINT_PLANE_PROBE_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-joint-plane-information-probe-closed) · [Copy provenance](../../../PROVENANCE.json)

