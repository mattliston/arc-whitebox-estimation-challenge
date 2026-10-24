# Correlation-graph features: both closed-loop development networks worsened

2026-09-05T06:07Z · experiment · rejected · measured

Held-out final one-step mean squared error (MSE): 5.68178323279156e-9 versus frozen parent's 5.703532246492671e-9, only 0.38% improvement. Matched full-final-feature closed loops, organizer 1e9 post means: - Rank 35: 2.0616038857016167e-8 versus 2.0444447453558874e-8 (+0.84%). - Rank 100: 2.065144495596603e-8 versus 2.00406278849e-8 (+3.05%). Failed the predeclared 5% improvement on each development network.

Recorded measurement:

    Held-out final one-step MSE: 5.68178323279156e-9 versus frozen parent's 5.703532246492671e-9, only 0.38% improvement. Matched full-final-feature closed loops, organizer 1e9 post means: - Rank 35: 2.0616038857016167e-8 versus 2.0444447453558874e-8 (+0.84%). - Rank 100: 2.065144495596603e-8 versus 2.00406278849e-8 (+3.05%). Failed the predeclared 5% improvement on each development network. No scalar-final refit, candidate port, 14-net panel, or submission. Preserve the 200 rows, model, payload, and receipts; do not rerun this family.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/chain_current_graph_training_v1.py](code/astra_phase2_reexamination_v1/chain_current_graph_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_radial_training.py](code/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/astra_phase2_reexamination_v1/generate_current_graph_rows_v1.py](code/astra_phase2_reexamination_v1/generate_current_graph_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_graph_features_v1.py](code/astra_phase2_reexamination_v1/radial_graph_features_v1.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/screen_current_graph_closedloop_v1.py](code/astra_phase2_reexamination_v1/screen_current_graph_closedloop_v1.py)
- [code/astra_phase2_reexamination_v1/screen_radial_closedloop.py](code/astra_phase2_reexamination_v1/screen_radial_closedloop.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [GRAPH_FEATURE_ROUND_V1_CLOSED.md#L1](GRAPH_FEATURE_ROUND_V1_CLOSED.md#L1)
- [GRAPH_FEATURE_ROUND_V1_CLOSED.md#L8](GRAPH_FEATURE_ROUND_V1_CLOSED.md#L8)
- [GRAPH_FEATURE_ROUND_V1_CLOSED.md#L12](GRAPH_FEATURE_ROUND_V1_CLOSED.md#L12)

<details>
<summary>All result files and records (2)</summary>

- [GRAPH_FEATURE_ROUND_V1_CLOSED.md](GRAPH_FEATURE_ROUND_V1_CLOSED.md) · record
- [GRAPH_FEATURE_ROUND_V1.md](GRAPH_FEATURE_ROUND_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-correlation-graph-features-both-closed-loop-development-networks-worsened) · [Copy provenance](../../../PROVENANCE.json)

