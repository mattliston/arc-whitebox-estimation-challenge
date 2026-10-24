# Fourth-cumulant deficit traced to uncarried newly generated joint fourth-order paths

2026-09-04 · experiment · informational · measured

Measured against the 32M truth (teammate E, T-44): card v right to 0.3%, k3 3.5->11% off, k4 = 0.99, 0.84, 0.71, 0.61, 0.54, 0.47, 0.42, 0.38 x truth at L1/3/5/7/9/11/13/15 on r35, r100 identical within 0.03; residual after a per-layer rescale 12-29%. Mechanism (estimator_cap.py:1071-1130, _cp4_reset): the fourth-cumulant canonical-polyadic carrier (CP4) carrier is capped at rank 2048 by top energy (lambda^2 ||u||^8); unit-vector births have tiny energy and are dropped first; the cap binds from layer 3; the dropped terms' DIAGONAL is subtracted from the carried k4. The phase-1 two-vertex law (k4 from k3) fails here (best-c error 52-81%): k4 must be carried.

Recorded measurement:

    Measured against the 32M truth (teammate E, T-44): card v right to 0.3%, k3 3.5->11% off, k4 = 0.99, 0.84, 0.71, 0.61, 0.54, 0.47, 0.42, 0.38 x truth at L1/3/5/7/9/11/13/15 on r35, r100 identical within 0.03; residual after a per-layer rescale 12-29%. Mechanism (estimator_cap.py:1071-1130, _cp4_reset): the CP4 carrier is capped at rank 2048 by top energy (lambda^2 ||u||^8); unit-vector births have tiny energy and are dropped first; the cap binds from layer 3; the dropped terms' DIAGONAL is subtracted from the carried k4. The phase-1 two-vertex law (k4 from k3) fails here (best-c error 52-81%): k4 must be carried. Since forcing true (v, k3, k4) in-card reaches the noise floor (T-43) and each state alone moves ~40% of the residual at the final read, this is the largest priced single defect on record. Tests (teammate A, urgent): zero-cost rescale k4 := card_k4 / s_l with s_l fit on r35 and frozen to r100 (F2J off/on, independent scoring); then the derived orphan-diagonal carry (push the dropped diagonal, 1024^2/layer) or the CP4 cap ladder 2048 -> 4096 -> 8192 with its bill. T-45..T-48: raising the CP4 cap to 4096/8192 makes k4 slightly WORSE (the cap is not the cause; the review's mechanism attribution is retracted, the defect stands - the scale loss is in the CP4 physics itself). The under-scaling is universal across the 8-net panel (panel-mean ratio 0.986 -> 0.424 at L1..L14, sd 0.005-0.018; k4_ratio_panel_8M.json). Zero-cost frozen rescale k4 := card_k4 / s_l: host -37.5% (r35) / -29.8% (r100), 65-85% of exact-k4's host gain; truth k3+k4 on the card's own v beats the deployed full card by 25-39% F2J-off. Cause search (teammate A) in progress; derived fix preferred. T-49..T-51: cause found (the card's fourth-order planes lose ~half of the (3,1)/(2,2) content born each layer from the fourth-order distinct classes; k4 tracks K22; 4-path words are +121-134% of kappa4, 3-stars -4%); derived born-word plane feed certified (block patterns carry 33%/49% of the path contribution) and priced (12 GEMM-eq/layer, ~27% util) - stays as the physics behind the table. Canonical table public_g19_attribution_v1/k4_scale_panel.json (panel mean 0.986 -> 0.403); on its own host 5.27e-8 (r35) / 5.00e-8 (r100) F2J off. T-71: the same defect on the PLANES K22 (0.92 -> 0.41 of truth, corr 0.96-0.99) and K31 (0.52 -> 0.30); frozen plane scale tables explode the card (co-adapted quartic star) - replacement, not scaling.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py](code/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py)
- [code/p2_width1024_depth16/public_deploy_composition_prep_v1/run_joint_refit.py](code/p2_width1024_depth16/public_deploy_composition_prep_v1/run_joint_refit.py)
- [code/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/build_gate1_supplies.py](code/p2_width1024_depth16/public_g19_attribution_v1/build_gate1_supplies.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/consumption_betas.py](code/p2_width1024_depth16/public_g19_attribution_v1/consumption_betas.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_c1_adjoint_rank.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_c1_adjoint_rank.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tangent_v2.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tangent_v2.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_g20_emission_retrofit_v1/estimator_retro.py](code/p2_width1024_depth16/public_g20_emission_retrofit_v1/estimator_retro.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/p2_width1024_depth16/public_g28_mech_features_v1/mechfeat15.py](code/p2_width1024_depth16/public_g28_mech_features_v1/mechfeat15.py)
- [code/p2_width1024_depth16/public_g29_bond_kernel_v1/k22_chain.py](code/p2_width1024_depth16/public_g29_bond_kernel_v1/k22_chain.py)
- [code/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py](code/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

</details>

## Evidence

- [REVIEW.md#L1](REVIEW.md#L1)
- [audit_alignment.log#L1](audit_alignment.log#L1)
- [LANE_g3_tracker.md#L577](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L577)

<details>
<summary>All result files and records (3)</summary>

- [REVIEW.md](REVIEW.md) · record
- [audit_alignment.log](audit_alignment.log) · log
- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-fourth-cumulant-deficit-traced-uncarried-newly-generated-joint-fourth) · [Copy provenance](../../../PROVENANCE.json)

