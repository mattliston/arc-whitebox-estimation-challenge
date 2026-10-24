# Bounded radial and factor-thinning variants failed their local accuracy gates

2026-09-05T00:52Z · experiment · informational · measured

- Signed-pair cap4096, honestly coadapted200rows, fixed160/40 split, C64,40epochs: closed-loop5.53288e-8 /5.40290e-8, versus full-bank learned 2.21231e-8 /2.11385e-8. Unlearned utilization.184676, but raw loss overwhelms saving. - Signed-pair cap4096, honestly coadapted200rows, fixed160/40 split, C64,40epochs: closed-loop5.53288e-8 /5.40290e-8, versus full-bank learned 2.21231e-8 /2.11385e-8.

Recorded measurement:

    - Signed-pair cap4096, honestly coadapted200rows, fixed160/40 split, C64,40epochs: closed-loop5.53288e-8 /5.40290e-8, versus full-bank learned 2.21231e-8 /2.11385e-8. Unlearned utilization.184676, but raw loss overwhelms saving. No panel or upload warranted. - Low-rank CP plane READ, retaining all factors and exact scalar diagonal: exactSVD rank64/128 both fail. Removing the diagonal before SVD does not fix it: rank1289.7903e-8 /1.1302e-7 versus full host2.7972e-8 /2.9453e-8. This closes this read-compression form, not an impossible plane transport. - Physical activation probability instead of standardized gate in GNN messages, same frozen200rows and40epochs: held-out one-step5.93605e-9 versus6.01057e-9, but closed-loop2.25120e-8 /2.15028e-8 is worse than original2.21231e-8 /2.11385e-8. Not promoted. Inference and trainer match. - Half precision has no billing advantage: installed official FlopScope.12 dtype rates give float16=float32=1, float64=2. No compute experiment run. - Complete binary cubic 3 lambda P^2 Q + kappa Q^3 represents negative diagonal remainders using the same two factor banks and two rank-N plane products. Algebra certificate passed. Full unlearned2.79799e-8 /2.95247e-8 is effectively unchanged versus two signed cubes. Discriminant clipping was not an important raw-error source on these two nets.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/build_direct_transform_v1.py](code/astra_phase2_reexamination_v1/build_direct_transform_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_direct_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_direct_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_v2.py](code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_v2.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_learned_d6_direct_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_learned_d6_direct_v1.py)
- [code/astra_phase2_reexamination_v1/chain_binary_compact_training_v1.py](code/astra_phase2_reexamination_v1/chain_binary_compact_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py](code/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/chain_radial_training.py](code/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/generate_binary_compact_rows.py](code/astra_phase2_reexamination_v1/generate_binary_compact_rows.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [code/astra_phase2_reexamination_v1/generate_compact_final_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_final_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_fullpair_final_rows_v1.py](code/astra_phase2_reexamination_v1/generate_fullpair_final_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_binary_cubic.py](code/astra_phase2_reexamination_v1/radial_binary_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_binary_gateprune.py](code/astra_phase2_reexamination_v1/radial_binary_gateprune.py)
- [code/astra_phase2_reexamination_v1/radial_binary_sparsecarry.py](code/astra_phase2_reexamination_v1/radial_binary_sparsecarry.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/screen_compact_final_v1.py](code/astra_phase2_reexamination_v1/screen_compact_final_v1.py)
- [code/astra_phase2_reexamination_v1/screen_fullpair_final_v1.py](code/astra_phase2_reexamination_v1/screen_fullpair_final_v1.py)
- [code/astra_phase2_reexamination_v1/train_compact_final_head_v1.py](code/astra_phase2_reexamination_v1/train_compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/train_fullpair_final_head_v1.py](code/astra_phase2_reexamination_v1/train_fullpair_final_head_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [RESEARCH_RECEIPTS_2026_09_04_LATE.md#L11](RESEARCH_RECEIPTS_2026_09_04_LATE.md#L11)
- [RESEARCH_RECEIPTS_2026_09_04_LATE.md#L24](RESEARCH_RECEIPTS_2026_09_04_LATE.md#L24)

<details>
<summary>All result files and records (1)</summary>

- [RESEARCH_RECEIPTS_2026_09_04_LATE.md](RESEARCH_RECEIPTS_2026_09_04_LATE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-bounded-radial-factor-thinning-variants-failed-their-local-accuracy) · [Copy provenance](../../../PROVENANCE.json)

