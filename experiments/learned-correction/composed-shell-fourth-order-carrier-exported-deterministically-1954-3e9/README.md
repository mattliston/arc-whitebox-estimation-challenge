# Composed shell and fourth-order carrier exported deterministically at 1954.3e9 FLOPs

2026-08-25 · experiment · informational · measured

The composed deployment card (no-eigh base + lambda3 shell + rank-4 carrier) builds byte-deterministically: billed 1954.3e9, margin 244.7e9 under 2^41, finite. The corrector retraining pipeline was FOUND live (trainer, exporter, and the frozen payload's source model - identity re-export reproduces the card byte-exactly) plus a new splice tool for integrated cards, self-checked by round-trip. Fire path: generate rows from this host, frozen J retrain, export, splice, smoke, meter, fire.

Recorded measurement:

    The composed deployment card (no-eigh base + lambda3 shell + rank-4 carrier) builds byte-deterministically: billed 1954.3e9, margin 244.7e9 under 2^41, finite. The corrector retraining pipeline was FOUND live (trainer, exporter, and the frozen payload's source model - identity re-export reproduces the card byte-exactly) plus a new splice tool for integrated cards, self-checked by round-trip. Fire path: generate rows from this host, frozen J retrain, export, splice, smoke, meter, fire.

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
<summary>Additional programs and dependencies (17)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/splice_payload.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/splice_payload.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/generate_supplied_row.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/generate_supplied_row.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/supply_core.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/supply_core.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/build_estimator_k22r.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/build_estimator_k22r.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/k22_refit_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/k22_refit_diagnostic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/score_pilot.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/score_pilot.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_native_pilot_v1/run_x_pilot.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_native_pilot_v1/run_x_pilot.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/estimator_x.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/estimator_x.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/instrumented_estimator_x.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/instrumented_estimator_x.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_supply_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_supply_card_v1/estimator.py)

</details>

## Evidence

- [BUILD_RECEIPT.json#L1](BUILD_RECEIPT.json#L1)
- [COMPOSITION_STATUS.json#L30](COMPOSITION_STATUS.json#L30)

<details>
<summary>All result files and records (2)</summary>

- [BUILD_RECEIPT.json](BUILD_RECEIPT.json) · result
- [COMPOSITION_STATUS.json](COMPOSITION_STATUS.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-composed-shell-fourth-order-carrier-exported-deterministically-1954-3e9) · [Copy provenance](../../../PROVENANCE.json)

