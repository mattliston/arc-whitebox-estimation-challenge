# Positive-path matched pilot closes without an expansion

2026-09-06T03:00Z · experiment · rejected · measured

Pooled final mean squared error (MSE) against existing1e9 truths on37/39/42/48: - Original bank reference: 2.12016561136e-8. - Refitted parent control: 2.23546349938e-8. - Refitted positive-path source: 2.18539989183e-8. - New/control .97760481996; new/bank1.03076848343;1/4 bank wins. Thus covariance Frobenius gains mostly miss the next value-bearing directions in this cell.

Recorded measurement:

    Pooled final MSE against existing1e9 truths on37/39/42/48: - Original bank reference: 2.12016561136e-8. - Refitted parent control: 2.23546349938e-8. - Refitted positive-path source: 2.18539989183e-8. - New/control .97760481996; new/bank1.03076848343;1/4 bank wins. The fixed .90/.92/3-wins gate fails. Both branches' final inputs, hosts and sigmas match their frozen on-policy training rows bytewise. No200-net expansion, alternate epoch/strength/age/source selection, panel or upload. The isolated FNP.12 birth-factor implementation is byte-identical to the NumPy algebra on its wide test. Its projected incremental15-layer bill is849,991,680 FLOPs, .17324% of the rank35 bank bill. Cost was not the reason for closure. The current official bank remains329926 at4.98223e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_positive_path_atomic_fnp_v1.py](code/astra_phase2_reexamination_v1/check_positive_path_atomic_fnp_v1.py)
- [code/astra_phase2_reexamination_v1/check_positive_path_value_transfer_v1.py](code/astra_phase2_reexamination_v1/check_positive_path_value_transfer_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/export_positive_path_pilot_prefix_v1.py](code/astra_phase2_reexamination_v1/export_positive_path_pilot_prefix_v1.py)
- [code/astra_phase2_reexamination_v1/generate_positive_path_pilot_final_row_v1.py](code/astra_phase2_reexamination_v1/generate_positive_path_pilot_final_row_v1.py)
- [code/astra_phase2_reexamination_v1/generate_positive_path_pilot_final_rows_v1.py](code/astra_phase2_reexamination_v1/generate_positive_path_pilot_final_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_positive_path_pilot_rows_v1.py](code/astra_phase2_reexamination_v1/generate_positive_path_pilot_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_positive_path_rows_v1.py](code/astra_phase2_reexamination_v1/generate_positive_path_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/judge_positive_path_pilot_row_v1.py](code/astra_phase2_reexamination_v1/judge_positive_path_pilot_row_v1.py)

<details>
<summary>Additional programs and dependencies (20)</summary>

- [code/astra_phase2_reexamination_v1/judge_positive_path_pilot_v1.py](code/astra_phase2_reexamination_v1/judge_positive_path_pilot_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py](code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py)
- [code/astra_phase2_reexamination_v1/positive_path_birth_reference_v1.py](code/astra_phase2_reexamination_v1/positive_path_birth_reference_v1.py)
- [code/astra_phase2_reexamination_v1/positive_path_final_reference_v1.py](code/astra_phase2_reexamination_v1/positive_path_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/positive_path_resummation_fnp_v1.py](code/astra_phase2_reexamination_v1/positive_path_resummation_fnp_v1.py)
- [code/astra_phase2_reexamination_v1/positive_path_resummation_v1.py](code/astra_phase2_reexamination_v1/positive_path_resummation_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py](code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py)
- [code/astra_phase2_reexamination_v1/train_positive_path_pilot_final_v1.py](code/astra_phase2_reexamination_v1/train_positive_path_pilot_final_v1.py)
- [code/astra_phase2_reexamination_v1/train_positive_path_pilot_prefix_v1.py](code/astra_phase2_reexamination_v1/train_positive_path_pilot_prefix_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [POSITIVE_PATH_MATCHED_PILOT_V1_CLOSED.md#L1](POSITIVE_PATH_MATCHED_PILOT_V1_CLOSED.md#L1)
- [POSITIVE_PATH_MATCHED_PILOT_V1_CLOSED.md#L10](POSITIVE_PATH_MATCHED_PILOT_V1_CLOSED.md#L10)

<details>
<summary>All result files and records (2)</summary>

- [POSITIVE_PATH_MATCHED_PILOT_V1_CLOSED.md](POSITIVE_PATH_MATCHED_PILOT_V1_CLOSED.md) · record
- [POSITIVE_PATH_MATCHED_PILOT_DESIGN_V1.md](../../error-attribution/positive-resummed-complete-paths-source-tests-pass-raw-gain/POSITIVE_PATH_MATCHED_PILOT_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-positive-path-matched-pilot-closes-without-expansion) · [Copy provenance](../../../PROVENANCE.json)

