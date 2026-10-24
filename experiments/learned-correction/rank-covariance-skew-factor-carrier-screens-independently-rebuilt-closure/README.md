# Rank, covariance, skew and factor-carrier screens on the independently rebuilt closure

2026-09-05T00:08Z · experiment · informational · measured

- Keep512 newborn CP terms, preserve dropped repeated entries in star: raw~9.9e-8 to1.36e-7, versus3.2e-8 full. Closed this pruning variant. - Gaussian Mehler order6:3.202e-8 /3.201e-8, effectively wash. - Full first-order Edgeworth pair corrections through correlation order2: 3.086e-8 /3.659e-8 with one-star host. - Keep512 newborn CP terms, preserve dropped repeated entries in star: raw~9.9e-8 to1.36e-7, versus3.2e-8 full.

Recorded measurement:

    - Keep512 newborn CP terms, preserve dropped repeated entries in star: raw~9.9e-8 to1.36e-7, versus3.2e-8 full. Closed this pruning variant. - Gaussian Mehler order6:3.202e-8 /3.201e-8, effectively wash. - Full first-order Edgeworth pair corrections through correlation order2: 3.086e-8 /3.659e-8 with one-star host. No consistent gain. - Fourth metric C+beta*m*m^T, beta=.25 or1:1.09e-7 to2.34e-7. Rejected. - Sym(C,D) fourth closure matching every partial trace: carry1 gives 3.109e-8 /3.016e-8; carry0 gives3.189e-8 /2.934e-8. Small gains only. - One cubic factor per birth became unstable. Rejected. - Two signed cubes per birth cancel the spurious pure-covariance cube algebraically and retain the leading coordinate-star component: analytic beta gives2.797e-8 /2.945e-8. Fitted beta worse3.137e-8 /3.263e-8. Current repeated entries remain exact; discriminant clipping leaves a residual star. Tiny algebra certificate passed. No learned head yet. - Compress only the subtractive cube component into top32/top64 covariance modes, restoring current repeated entries:1e-6 to1e-5 raw. Rejected. - Keep all coordinate stars in the covariance-metric host (accuracy-only, roughly double carrier):2.585e-8 /2.601e-8. - All-star plus full Edgeworth pair map:2.280e-8 /2.861e-8. Two-cube plus full pair map:2.419e-8 /3.029e-8. No consistent late gain. - Natural K4-to-K3 quadratic-vertex source, represented by an extra covariance-vector cubic plus corrected CP birth coefficient: 3.146e-8 /2.890e-8. Modest gain, not coadapted or metered yet. There is no claimed universal impossibility. These receipts close the tested implementations, not all representations. The ~4e-9 goal is still unmet.

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

- [RESEARCH_RECEIPTS_2026_09_04.md#L48](../../wall-and-residual/exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L48)
- [RESEARCH_RECEIPTS_2026_09_04.md#L57](../../wall-and-residual/exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L57)
- [RESEARCH_RECEIPTS_2026_09_04.md#L1](../../wall-and-residual/exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L1)

<details>
<summary>All result files and records (1)</summary>

- [RESEARCH_RECEIPTS_2026_09_04.md](../../wall-and-residual/exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-rank-covariance-skew-factor-carrier-screens-independently-rebuilt-closure) · [Copy provenance](../../../PROVENANCE.json)

