# Residual fourth-cumulant bridge fails the actual value gate

2026-09-06T03:43Z · experiment · rejected · measured

V1 stopped before results because a full physical tensor was compared to an active-only bank plane. V2 applied the actual active support and stopped on a1.0409154e-6 F64-physical-versus-F32-query difference. The preregistered <=.75 next-mean ratio for both oracle sources AND the exact physical source fails.

Recorded measurement:

    V1 stopped before results because a full physical tensor was compared to an active-only bank plane. V2 applied the actual active support and stopped on a1.0409154e-6 F64-physical-versus-F32-query difference. V3 did not loosen that check: it replayed both saved objects independently, certifying the physical tensor slices/plane and actual consumer state byte-exactly. The physical/query precision difference is reported, not an accuracy gain. The full projected residual is masked to current active support, retaining all diagonal reads. No source coefficient or gain gate was changed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_one_high_actual_value_v1.py](code/astra_phase2_reexamination_v1/check_one_high_actual_value_v1.py)
- [code/astra_phase2_reexamination_v1/check_one_high_actual_value_v2.py](code/astra_phase2_reexamination_v1/check_one_high_actual_value_v2.py)
- [code/astra_phase2_reexamination_v1/check_one_high_actual_value_v3.py](code/astra_phase2_reexamination_v1/check_one_high_actual_value_v3.py)
- [code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py](code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/qualify_one_high_boundary_v1.py](code/astra_phase2_reexamination_v1/qualify_one_high_boundary_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py](code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [ONE_HIGH_ACTUAL_VALUE_V3_CLOSED.md#L1](ONE_HIGH_ACTUAL_VALUE_V3_CLOSED.md#L1)

<details>
<summary>All result files and records (3)</summary>

- [ONE_HIGH_ACTUAL_VALUE_V3_CLOSED.md](ONE_HIGH_ACTUAL_VALUE_V3_CLOSED.md) · record
- [one_high_actual_value_r35_l12_v3.json](one_high_actual_value_r35_l12_v3.json) · result
- [ONE_HIGH_ACTUAL_VALUE_DESIGN_V1.md](ONE_HIGH_ACTUAL_VALUE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-residual-fourth-cumulant-bridge-fails-actual-value-gate) · [Copy provenance](../../../PROVENANCE.json)

