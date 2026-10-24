# Corrector retraining on the joint host: adjusted 2.833e-8 lost to the 2.2843e-8 submission

2026-08-25 · experiment · informational · measured

as the deployed-frame confirmation: the per-host retraining pipeline executed end to end on the joint host (frozen 160/40 retrain, export, splice into the composed card), and the deployed-frame projection came in at 2.833e-8 adjusted in-sample-optimistic - WORSE than the 2.2843e-8 bank, verdict CONFIRMATION_NO_FIRE, exactly as the flat joint floor predicted. The pipeline itself is proven and stands ready for any host that actually moves the floor.

Recorded measurement:

    SOLVED as the deployed-frame confirmation: the per-host retraining pipeline executed end to end on the joint host (frozen 160/40 retrain, export, splice into the composed card), and the deployed-frame projection came in at 2.833e-8 adjusted in-sample-optimistic - WORSE than the 2.2843e-8 bank, verdict CONFIRMATION_NO_FIRE, exactly as the flat joint floor predicted. The pipeline itself is proven and stands ready for any host that actually moves the floor.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v3/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v3/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/build_composed.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/build_composed.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/build_shell_supply.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/build_shell_supply.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/estimator_composed.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/estimator_composed.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/estimator_composed_refit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/estimator_composed_refit.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/estimator_shell_supply.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/estimator_shell_supply.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_joint_refit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_joint_refit.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_one_forward.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_one_forward.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_shell_supply_eval.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_shell_supply_eval.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_supplied_refit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_supplied_refit.py)

<details>
<summary>Additional programs and dependencies (14)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/splice_payload.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/splice_payload.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/generate_supplied_row.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/generate_supplied_row.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/supply_core.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/supply_core.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/build_estimator_k22r.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/build_estimator_k22r.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/estimator_x.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/estimator_x.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/instrumented_estimator_x.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/instrumented_estimator_x.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_supply_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_supply_card_v1/estimator.py)

</details>

## Evidence

- [JOINT_REFIT_DEPLOYED.json#L1](JOINT_REFIT_DEPLOYED.json#L1)
- [BUILD_RECEIPT.json#L1](../composed-shell-fourth-order-carrier-exported-deterministically-1954-3e9/BUILD_RECEIPT.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [JOINT_REFIT_DEPLOYED.json](JOINT_REFIT_DEPLOYED.json) · result
- [BUILD_RECEIPT.json](../composed-shell-fourth-order-carrier-exported-deterministically-1954-3e9/BUILD_RECEIPT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-corrector-retraining-joint-host-adjusted-2-833e-8-lost) · [Copy provenance](../../../PROVENANCE.json)

