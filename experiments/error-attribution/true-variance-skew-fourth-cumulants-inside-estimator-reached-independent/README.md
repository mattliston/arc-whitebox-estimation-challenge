# True variance, skew and fourth cumulants inside the estimator reached the independent-truth noise floor

2026-09-04 · experiment · informational · measured

T-43 (frame-free): forcing 32M-truth per-layer v, k3, k4 into the card's OWN recursion puts the host at the independent-draw noise floor on r35 (-1.4e-9) and r100 (+1.9e-9) vs deployed 2.7e-8; mean and off-diagonal covariance add nothing; v alone removes 18% of the host error, v+k3+k4 ~100% (joint, nonlinear). The post-hoc linear consumption frame (whose ceilings read 37-68%) is retired for value. Card today: v 0.3-1% error, k3 6-11%, k4 unmeasured; engine-class carry k3 4-6%; cum4 (2,1,1) feed cuts k3 error ~15% at L8-14.

Recorded measurement:

    T-43 (frame-free): forcing 32M-truth per-layer v, k3, k4 into the card's OWN recursion puts the host at the independent-draw noise floor on r35 (-1.4e-9) and r100 (+1.9e-9) vs deployed 2.7e-8; mean and off-diagonal covariance add nothing; v alone removes 18% of the host error, v+k3+k4 ~100% (joint, nonlinear). The post-hoc linear consumption frame (whose ceilings read 37-68%) is retired for value. Card today: v 0.3-1% error, k3 6-11%, k4 unmeasured; engine-class carry k3 4-6%; cum4 (2,1,1) feed cuts k3 error ~15% at L8-14. Running (teammate A): truth + controlled noise per state and truth + s*(host - truth), F2J off, independent scoring, r35/r100 -> the accuracy each state needs for Matt's bank (-49%) and the goal (-66%). 32M independent scoring draws generating. T-44 (review): instruments cleared of coding bugs; ranked in-card tests: price states by window; k4 cap fix; CP4 cap ladder; skew fixes together with the k4 fix; F2J refit on MPS with corrected states as tape features; engine k3u in-card via gate-only carry. Seed caution: the 32M truth contains the 8M draw and indep32M contains indep4M. LADDER (T-45/T-47, both nets, indep32M frame): truth (v,k3,k4) -> 1.6-1.7e-9 (93-94% cut); structured halving of the card's real errors -> 2.0-2.25e-8 (beats the deployed card 18-22% F2J-off, +32% F2J-on on r35), quartering -> 5.3-6.1e-9 (goal); iid v noise 0.1% erases the gain (v supplies must be structured or near-exact); k3 tolerates ~2-4%, k4 ~10%. T-56 position: two frozen tables halve the host error on both nets; next (teammate A): attribution of the remaining host error among v / k3-residual / k4-residual, and whether v or the k4 residual has a frozen-table-able universal structure. NIGHT VERDICT (T-63): every cheap route closed with receipts; the exact open problem is N0.1.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py](code/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py)
- [code/p2_width1024_depth16/public_deploy_composition_prep_v1/run_joint_refit.py](code/p2_width1024_depth16/public_deploy_composition_prep_v1/run_joint_refit.py)
- [code/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/build_gate1_supplies.py](code/p2_width1024_depth16/public_g19_attribution_v1/build_gate1_supplies.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/consumption_betas.py](code/p2_width1024_depth16/public_g19_attribution_v1/consumption_betas.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tangent_v2.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tangent_v2.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4_g24fix.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4_g24fix.py)
- [code/p2_width1024_depth16/public_g20_emission_retrofit_v1/estimator_retro.py](code/p2_width1024_depth16/public_g20_emission_retrofit_v1/estimator_retro.py)

<details>
<summary>Additional programs and dependencies (30)</summary>

- [code/p2_width1024_depth16/public_g20_emission_retrofit_v1/perp_judge_engine.py](code/p2_width1024_depth16/public_g20_emission_retrofit_v1/perp_judge_engine.py)
- [code/p2_width1024_depth16/public_g20_emission_retrofit_v1/run_retro_screen.py](code/p2_width1024_depth16/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/p2_width1024_depth16/public_g21_pair_consumer_v1/mc_final_indep.py](code/p2_width1024_depth16/public_g21_pair_consumer_v1/mc_final_indep.py)
- [code/p2_width1024_depth16/public_g21_pair_consumer_v1/pairmap.py](code/p2_width1024_depth16/public_g21_pair_consumer_v1/pairmap.py)
- [code/p2_width1024_depth16/public_g21_pair_consumer_v1/run_pair_consumer.py](code/p2_width1024_depth16/public_g21_pair_consumer_v1/run_pair_consumer.py)
- [code/p2_width1024_depth16/public_g22_operator_features_v1/gen_rows_mech.py](code/p2_width1024_depth16/public_g22_operator_features_v1/gen_rows_mech.py)
- [code/p2_width1024_depth16/public_g22_operator_features_v1/jext/run_experiment_ext.py](code/p2_width1024_depth16/public_g22_operator_features_v1/jext/run_experiment_ext.py)
- [code/p2_width1024_depth16/public_g22_operator_features_v1/run_refit_ext.py](code/p2_width1024_depth16/public_g22_operator_features_v1/run_refit_ext.py)
- [code/p2_width1024_depth16/public_g22_operator_features_v1/score_refit_gate_ext.py](code/p2_width1024_depth16/public_g22_operator_features_v1/score_refit_gate_ext.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_cum4_feed.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_cum4_feed.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_cum4_feed2.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_cum4_feed2.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_cum4_feed2_sp.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_cum4_feed2_sp.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_pathfeed_patterns.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_pathfeed_patterns.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_star4.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_star4.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_triple_gh.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/cert_triple_gh.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/k31_channel_check.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/k31_channel_check.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/k31_target_check.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/k31_target_check.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/k31_target_check_g24fix.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/k31_target_check_g24fix.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/k3gen.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/k3gen.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/k4_onestep_check.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/k4_onestep_check.py)
- [code/p2_width1024_depth16/public_g23_k3_supply_v1/run_k3gen_layers.py](code/p2_width1024_depth16/public_g23_k3_supply_v1/run_k3gen_layers.py)
- [code/p2_width1024_depth16/public_g27_k3_structure_v1/k3_structure.py](code/p2_width1024_depth16/public_g27_k3_structure_v1/k3_structure.py)
- [code/p2_width1024_depth16/public_g27_k3_structure_v1/state_structure.py](code/p2_width1024_depth16/public_g27_k3_structure_v1/state_structure.py)
- [code/p2_width1024_depth16/public_g28_mech_features_v1/mechfeat.py](code/p2_width1024_depth16/public_g28_mech_features_v1/mechfeat.py)
- [code/p2_width1024_depth16/public_g28_mech_features_v1/mechfeat15.py](code/p2_width1024_depth16/public_g28_mech_features_v1/mechfeat15.py)
- [code/p2_width1024_depth16/public_g28_mech_features_v1/ridge_study.py](code/p2_width1024_depth16/public_g28_mech_features_v1/ridge_study.py)
- [code/p2_width1024_depth16/public_g29_bond_kernel_v1/k22_chain.py](code/p2_width1024_depth16/public_g29_bond_kernel_v1/k22_chain.py)
- [code/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py](code/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

</details>

## Evidence

- [LANE.md#L134](LANE.md#L134)
- [LANE_g3_tracker.md#L574](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L574)

<details>
<summary>All result files and records (2)</summary>

- [LANE.md](LANE.md) · record
- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-true-variance-skew-fourth-cumulants-inside-estimator-reached-independent) · [Copy provenance](../../../PROVENANCE.json)

