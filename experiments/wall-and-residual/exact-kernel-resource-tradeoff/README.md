# Exact-kernel resource tradeoff

2026-09-05T00:08Z · experiment · informational · measured

Depth-six rectangular products with 8192-wide tiles preserve the full panel: public8 adjusted7.798e-9, virgin6 adjusted6.952e-9, utilization.3115. Actual client/server utilization.311646 and residual.22165s passed, but native 35/100/35 peak resident-set size (RSS) reached9.433GB, above8GiB. Depth-six rectangular products with 8192-wide tiles preserve the full panel: public8 adjusted7.798e-9, virgin6 adjusted6.952e-9, utilization.3115.

Recorded measurement:

    Depth-six rectangular products with 8192-wide tiles preserve the full panel: public8 adjusted7.798e-9, virgin6 adjusted6.952e-9, utilization.3115. Actual client/server utilization.311646 and residual.22165s passed, but native 35/100/35 peak RSS reached9.433GB, above8GiB. Do not submit this version. Halving tiles to4096 costs only utilization.312745 at the client boundary. Native head parity and byte-identical repeated output pass, peak RSS7.373GB. Under concurrent load its actual client residual was.42258s, above.4s; do not declare a complete resource pass from native results alone. Depth-five, 8192-wide tiles are the current resource-safe alternative. Actual client/server utilization.322554, wall49.14s, residual.33626s under concurrent load. Native first-call peak RSS7.229GB, learned-delta maximum deviation6.64e-8 versus NumPy. Final repeated-call and 14-net checks ongoing at the time of this entry. The older D4 full-bank candidate itself reached 8.914GB peak native RSS, so it is not an unconditional resource-safe fallback.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/attribute_radial_rows.py](code/astra_phase2_reexamination_v1/attribute_radial_rows.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_learned_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_learned_v1.py)
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

- [RESEARCH_RECEIPTS_2026_09_04.md#L113](RESEARCH_RECEIPTS_2026_09_04.md#L113)
- [RESEARCH_RECEIPTS_2026_09_04.md#L1](RESEARCH_RECEIPTS_2026_09_04.md#L1)
- [RESEARCH_RECEIPTS_2026_09_04.md#L130](RESEARCH_RECEIPTS_2026_09_04.md#L130)

<details>
<summary>All result files and records (1)</summary>

- [RESEARCH_RECEIPTS_2026_09_04.md](RESEARCH_RECEIPTS_2026_09_04.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-exact-kernel-resource-tradeoff) · [Copy provenance](../../../PROVENANCE.json)

