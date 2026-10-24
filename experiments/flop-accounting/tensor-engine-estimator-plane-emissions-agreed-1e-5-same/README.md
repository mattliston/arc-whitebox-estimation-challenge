# Tensor-engine and estimator plane emissions agreed to 1e-5 on the same state

2026-09-02 · experiment · rejected · measured

Engine: Mehler stack + Edgeworth channels on m,v,k3,k4,C,third-cumulant repeated-index plane (K21),fourth-cumulant two-pair plane (K22),fourth-cumulant triple-single plane (K31) (run_tensor_recursion_v4.py:59-124). Card: _gaussian_pair moments -> _connected_planes + _cp_post_read (inherited + birth words). Fix path: retrofit into the card (cost ~0 if it rides the card's dense planes; coadaptation risk measured with fitted transport-energy residual correction (F2J) on/off) else the parallel delta chain (SUB-2).

Recorded measurement:

    Engine: Mehler stack + Edgeworth channels on m,v,k3,k4,C,K21,K22,K31 (run_tensor_recursion_v4.py:59-124). Card: _gaussian_pair moments -> _connected_planes + _cp_post_read (inherited + birth words). Fix path: retrofit into the card (cost ~0 if it rides the card's dense planes; coadaptation risk measured with F2J on/off) else the parallel delta chain (SUB-2). Priced by teammate A (P2R0G19B_PKG_SUB1_EMISSION_RETROFIT). T-28 (teammate A): on the card's own state the engine's emission_K21 and the card's plane target agree to ~1e-5 rms off-diagonal, 50x below the truth-card gap (5.5e-4), uncorrelated with it. Same nine Edgeworth channels, same connected formula; no input missing. Not the mechanism; the direct emission-delta sidecar reads -0.1..-1.3% (r35) / 0.0% (r100).

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
- [code/p2_width1024_depth16/public_g20_emission_retrofit_v1/build_sidecar_planes.py](code/p2_width1024_depth16/public_g20_emission_retrofit_v1/build_sidecar_planes.py)
- [code/p2_width1024_depth16/public_g20_emission_retrofit_v1/cert_resid_cp.py](code/p2_width1024_depth16/public_g20_emission_retrofit_v1/cert_resid_cp.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/p2_width1024_depth16/public_g20_emission_retrofit_v1/estimator_retro.py](code/p2_width1024_depth16/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/p2_width1024_depth16/public_g20_emission_retrofit_v1/run_retro_screen.py](code/p2_width1024_depth16/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/p2_width1024_depth16/public_g28_mech_features_v1/mechfeat15.py](code/p2_width1024_depth16/public_g28_mech_features_v1/mechfeat15.py)
- [code/p2_width1024_depth16/public_g29_bond_kernel_v1/k22_chain.py](code/p2_width1024_depth16/public_g29_bond_kernel_v1/k22_chain.py)
- [code/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py](code/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

</details>

## Evidence

- [LANE.md#L131](LANE.md#L131)

<details>
<summary>All result files and records (1)</summary>

- [LANE.md](LANE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-02-tensor-engine-estimator-plane-emissions-agreed-1e-5-same) · [Copy provenance](../../../PROVENANCE.json)

