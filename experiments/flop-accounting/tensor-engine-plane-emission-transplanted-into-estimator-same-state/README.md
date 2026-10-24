# Tensor-engine plane emission transplanted into the estimator: same-state formula comparison

2026-09-02 · experiment · rejected · measured

Teammate A (fork 'emission-retrofit'): term-level diff of the engine's validated emission_K21 (Mehler stack + Edgeworth on m,v,k3,k4,C,third-cumulant repeated-index plane (K21),fourth-cumulant two-pair plane (K22),fourth-cumulant triple-single plane (K31)) vs the card's _gaussian_pair -> _connected_planes + _cp_post_read; retrofit into a COPY of the card (public_g20_emission_retrofit_v1), certified per term; raw on r35/r100 with fitted transport-energy residual correction (F2J) on/off; billed-flop delta. Prices mechanism leaves N1.2a/N1.2b (P2R0G19B_MECH_N1_2A_EMISSION, _N1_2B_PUSHREAD) and N4. RESULT (T-25/T-28, public_g20_emission_retrofit_v1): emission null; mechanism = carried pushed residual; in-card carry is net-negative with F2J on and +17% util; realizing it needs F2J refit + a >=3x cheaper carrier (rank, not rows).

Recorded measurement:

    Teammate A (fork 'emission-retrofit'): term-level diff of the engine's validated emission_K21 (Mehler stack + Edgeworth on m,v,k3,k4,C,K21,K22,K31) vs the card's _gaussian_pair -> _connected_planes + _cp_post_read; retrofit into a COPY of the card (public_g20_emission_retrofit_v1), certified per term; raw on r35/r100 with F2J on/off; billed-flop delta. Prices mechanism leaves N1.2a/N1.2b (P2R0G19B_MECH_N1_2A_EMISSION, _N1_2B_PUSHREAD) and N4. RESULT (T-25/T-28, public_g20_emission_retrofit_v1): emission null; mechanism = carried pushed residual; in-card carry is net-negative with F2J on and +17% util; realizing it needs F2J refit + a >=3x cheaper carrier (rank, not rows). Closed as measured.

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

- [LANE.md#L1](../tensor-engine-estimator-plane-emissions-agreed-1e-5-same/LANE.md#L1)

<details>
<summary>All result files and records (1)</summary>

- [LANE.md](../tensor-engine-estimator-plane-emissions-agreed-1e-5-same/LANE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-02-tensor-engine-plane-emission-transplanted-into-estimator-same-state) · [Copy provenance](../../../PROVENANCE.json)

