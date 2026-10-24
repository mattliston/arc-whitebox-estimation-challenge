# Centered-feedback convention closed after matched closed-loop test

2026-09-05T15:32Z · experiment · rejected · measured

September5,2026. Both honest two-net screens passed (-18.7%/-23.3% late one-step mean squared error (MSE)). Inference, not an additional experiment: centered-covariance feedback also changes sensitivity to upstream mean error.

Recorded measurement:

    September5,2026. Both honest two-net screens passed (-18.7%/-23.3% late one-step MSE). The full200-row fixed40-epoch matched fit was healthy and reduced valid40 final one-step MSE from parent5.70353225e-9 to 4.50889290e-9 (-20.95%). This teacher-forced improvement did NOT survive materially in the learned closed loop. Stored1e9 final post means, same full-final-feature comparisons: rank35 parent2.04444474536e-8 -> centered2.00931149812e-8, ratio.9828153; rank100 parent2.00406278849e-8 -> centered1.97014567293e-8, ratio.9830758. Only -1.7185%/-1.6924%, below the predeclared8%-on-each gate. No scalar-final refit, candidate port, panel, or submission. No feedback coefficient sweep/damping. All200 rows, checkpoint, payload and screens are preserved under their new names. Official bank329926 remains4.98223146018e-9 adjusted and unmodified.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/chain_centered_feedback_training_v1.py](code/astra_phase2_reexamination_v1/chain_centered_feedback_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_radial_training.py](code/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/astra_phase2_reexamination_v1/generate_centered_feedback_rows_v1.py](code/astra_phase2_reexamination_v1/generate_centered_feedback_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_centered_feedback_v1.py](code/astra_phase2_reexamination_v1/radial_centered_feedback_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/astra_phase2_reexamination_v1/screen_centered_feedback_closedloop_v1.py](code/astra_phase2_reexamination_v1/screen_centered_feedback_closedloop_v1.py)
- [code/astra_phase2_reexamination_v1/screen_radial_closedloop.py](code/astra_phase2_reexamination_v1/screen_radial_closedloop.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [CENTERED_FEEDBACK_ROUND_V1_CLOSED.md#L1](CENTERED_FEEDBACK_ROUND_V1_CLOSED.md#L1)
- [CENTERED_FEEDBACK_ROUND_V1_CLOSED.md#L6](CENTERED_FEEDBACK_ROUND_V1_CLOSED.md#L6)
- [CENTERED_FEEDBACK_ROUND_V1_CLOSED.md#L12](CENTERED_FEEDBACK_ROUND_V1_CLOSED.md#L12)

<details>
<summary>All result files and records (2)</summary>

- [CENTERED_FEEDBACK_ROUND_V1_CLOSED.md](CENTERED_FEEDBACK_ROUND_V1_CLOSED.md) · record
- [CENTERED_FEEDBACK_ROUND_V1.md](CENTERED_FEEDBACK_ROUND_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-centered-feedback-convention-closed-after-matched-closed-loop-test) · [Copy provenance](../../../PROVENANCE.json)

