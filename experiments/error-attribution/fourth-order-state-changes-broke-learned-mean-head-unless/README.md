# Fourth-order state changes broke the learned mean head unless the two were co-adapted

2026-09-04 · experiment · informational · measured

fourth-cumulant two-pair plane (K22)-chain co-adaptation DEFERRED: rows with the numpy chain active take ~10+ min each (6/200 after 25 min = ~6 h for the set); low prior (the chain's plane fix was 84% in-span of the patch in the open-loop era) and an undeployable numpy chain -> killed to free the CPU for the cheap-card runs. Resume only if the utilization line fails: launch_mech15_k22.sh + chain_k22_coadapt.sh.

Recorded measurement:

    - T-117 PLANE SCALE (beta) IN THE HONEST LOOP r35: 9.96e-5 - DIVERGES (4000x). The frozen K22/K31 scale table is not usable inside the closed loop as is (numerical blow-up, not mere co-adaptation). Ratio mode / r100 and the engine K22 chain pending. HVBIL final (1e9): mix 0.5 3.10/4.61, mix 1.0 6.07/8.89: dead.
    
    - T-117 RESULTS: plane scale (both modes, both nets) DIVERGES in the loop (1e-4 class); engine K22 chain in-card, un-adapted head, r35 6.13e-8 (8M; loop alone 2.49e-8) -> a deterministic source fix needs co-adaptation of the mean head to be judged. T-117e second-stage probe (cross-layer ridge on the loop's own rows, 816 features): 17% of the pre-final-correction residual, of which 15% from the final layer's own features -> no cross-layer information; dead (consistent with T-112). T-117f mean-head gap: train 67.9% vs valid 66.9% -> no model variance, ensembling useless; pure information limit. T-118 PIVOT: utilization (0.61 vs the competitor's 0.22) is the structural gap - at util 0.22 our raw 2.5e-8 would adjust to 5.5e-9. Test a cheaper card inside the honest loop (CP3 cap 8192 = half the default 16384; CP4 cap 1024/512 vs 2048), raw x util, then co-adapt the mean head to the cheaper card if the product improves (cheap_card_in_loop.sh). Also queued: co-adaptation to the K22 chain (rows with the chain active, generator variant).
    
    - T-117d k4 RESCALE (deterministic per-layer table, full, un-adapted head) r35 7.25e-8 (1e9): 3x worse (true k4 half was +20%). Every state change breaks the co-adapted mean head; only co-adapted comparisons are meaningful from here. Running: T-117g K22-chain co-adaptation (rows_meanleak_mech15_k22 generating with the chain active -> GNN_HK22 -> loop with the chain), T-118 cheap-card runs (CP3 cap 8192, CP4 cap 1024/512; raw x util).
    
    - T-117g K22-chain co-adaptation DEFERRED: rows with the numpy chain active take ~10+ min each (6/200 after 25 min = ~6 h for the set); low prior (the chain's plane fix was 84% in-span of the patch in the open-loop era) and an undeployable numpy chain -> killed to free the CPU for the cheap-card runs. Resume only if the utilization line fails: launch_mech15_k22.sh + chain_k22_coadapt.sh.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/cheap_card_in_loop.sh](code/public_g22_operator_features_v1/cheap_card_in_loop.sh)
- [code/public_g22_operator_features_v1/launch_mech15_k22.sh](code/public_g22_operator_features_v1/launch_mech15_k22.sh)
- [code/public_g22_operator_features_v1/chain_k22_coadapt.sh](code/public_g22_operator_features_v1/chain_k22_coadapt.sh)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_k22.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_k22.py)
- [code/public_g22_operator_features_v1/run_gen_meanleak_mech15_k22.py](code/public_g22_operator_features_v1/run_gen_meanleak_mech15_k22.py)
- [code/public_g22_operator_features_v1/score_vs_truthpost.py](code/public_g22_operator_features_v1/score_vs_truthpost.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L876](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L876)
- [LANE_g3_tracker.md#L877](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L877)
- [LANE_g3_tracker.md#L878](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L878)
- [LANE_g3_tracker.md#L879](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L879)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-fourth-order-state-changes-broke-learned-mean-head-unless) · [Copy provenance](../../../PROVENANCE.json)

