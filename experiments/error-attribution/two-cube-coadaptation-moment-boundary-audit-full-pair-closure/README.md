# Two-cube coadaptation and moment-boundary audit of the full-pair closure

2026-09-05T00:08Z · experiment · informational · measured

The two-cube analytic-birth host was coadapted on 200 honest rows using the fixed 160/40 cross-net split, C64, 40 epochs, mean head only. Held-out final one-step mean squared error (MSE) is 6.01057e-9 versus host 7.685e-9 (21.79% removal). - 329790, recovered historical third-cumulant canonical-polyadic carrier (CP3)-8192 payload: adjusted1.4369934e-8, raw3.1560041e-8, 10.84% better than official329754.

Recorded measurement:

    The two-cube analytic-birth host was coadapted on 200 honest rows using the fixed 160/40 cross-net split, C64, 40 epochs, mean head only. Held-out final one-step MSE is 6.01057e-9 versus host 7.685e-9 (21.79% removal). Closed-loop development raw is 2.21231e-8 / 2.11385e-8, against the stored 1e9 means. Payload: payload_binary_radial_v1.npz; rows: rows_binary_radial_v1. The full independent4M noise-corrected panel for candidate_binary_learned_v1.py gives public8 raw2.503e-8, utilization.3407, adjusted8.527e-9; virgin6 raw2.231e-8, utilization.3407, adjusted7.601e-9. This is a local competitive result, not an official graded score. The diagonal-only covariance feedback variant was regenerated and retrained honestly, not tested with an unmatched head. Held-out one-step final MSE is 4.79260e-9 versus host5.774e-9, but closed-loop development raw is 2.27396e-8 / 2.04621e-8: mixed versus the full-feedback model. Not promoted. - 329790, recovered historical CP3-8192 payload: adjusted1.4369934e-8, raw3.1560041e-8, 10.84% better than official329754. Current official best. - 329798, initial new radial learned card: adjusted1.7476524e-8, raw3.1709440e-8. Its official utilization was .55115, not local .3408. Root cause was our native-only dtype gate: remote .dtype is a string, so comparing it directly with fnp.float32 disabled every fast product. All newer learned binary candidates compare str(dtype) instead. The fixed radial candidate passed the actual numpy-free client/server gate at utilization.340831, residual.16444s. See the separate dtype-bug receipt.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/candidate_binary_learned_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_learned_v1.py)
- [code/astra_phase2_reexamination_v1/attribute_radial_rows.py](code/astra_phase2_reexamination_v1/attribute_radial_rows.py)
- [code/astra_phase2_reexamination_v1/candidate_radial_gnn_splitread_v1.py](code/astra_phase2_reexamination_v1/candidate_radial_gnn_splitread_v1.py)
- [code/astra_phase2_reexamination_v1/chain_binary_training_v1.py](code/astra_phase2_reexamination_v1/chain_binary_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_radial_training.py](code/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/astra_phase2_reexamination_v1/generate_binary_rows.py](code/astra_phase2_reexamination_v1/generate_binary_rows.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_cubic.py](code/astra_phase2_reexamination_v1/radial_binary_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [RESEARCH_RECEIPTS_2026_09_04.md#L77](../../wall-and-residual/exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L77)
- [RESEARCH_RECEIPTS_2026_09_04.md#L81](../../wall-and-residual/exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L81)
- [RESEARCH_RECEIPTS_2026_09_04.md#L1](../../wall-and-residual/exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L1)
- [RESEARCH_RECEIPTS_2026_09_04.md#L99](../../wall-and-residual/exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L99)

<details>
<summary>All result files and records (1)</summary>

- [RESEARCH_RECEIPTS_2026_09_04.md](../../wall-and-residual/exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-two-cube-coadaptation-moment-boundary-audit-full-pair-closure) · [Copy provenance](../../../PROVENANCE.json)

