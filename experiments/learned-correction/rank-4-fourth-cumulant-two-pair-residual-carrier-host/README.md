# Rank-4 fourth-cumulant two-pair residual carrier: host error fell 35.6%, corrected error stayed flat

2026-08-25 · experiment · informational · measured

The transport-fidelity probe proved the deployed quartic star push exact on its carried family while the fourth-cumulant two-pair plane (K22)-residual channel (computed at every reset, alpha-fitted, rest discarded) is transported at exactly zero. The rank-4 carrier with an exact-on-family push (independently audited; the first build's k31 channel was exactly doubled and was caught pre-data) measured, in a sealed 200-row pilot: host raw 9.039e-8 -> 5.826e-8, minus 35.6 percent, 200 of 200 rows, t=-59.9. The refit-J diagnostic showed the additive invariant EXACT (corrected 3.133e-8 vs 3.124e-8): the carrier removed almost purely tape-reconstructable error.

Recorded measurement:

    The transport-fidelity probe proved the deployed quartic star push exact on its carried family while the K22-residual channel (computed at every reset, alpha-fitted, rest discarded) is transported at exactly zero. The rank-4 carrier with an exact-on-family push (independently audited; the first build's k31 channel was exactly doubled and was caught pre-data) measured, in a sealed 200-row pilot: host raw 9.039e-8 -> 5.826e-8, minus 35.6 percent, 200 of 200 rows, t=-59.9. The refit-J diagnostic showed the additive invariant EXACT (corrected 3.133e-8 vs 3.124e-8): the carrier removed almost purely tape-reconstructable error.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v3/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v3/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_redesign_study_v1/k22_carrier_billing_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_redesign_study_v1/k22_carrier_billing_probe.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_redesign_study_v1/star_spectrum_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_redesign_study_v1/star_spectrum_probe.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_redesign_study_v1/star_transport_fidelity_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_redesign_study_v1/star_transport_fidelity_probe.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/build_estimator_k22r.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/build_estimator_k22r.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/run_ab_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/run_ab_probe.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/generate_pilot_row.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/generate_pilot_row.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/k22_refit_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/k22_refit_diagnostic.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/run_generation.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/run_generation.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/score_pilot.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_pilot_v1/score_pilot.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/instrumented_estimator_x.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_lambda3_shell_probe_v1/instrumented_estimator_x.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_repeatability_v3/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_repeatability_v3/estimator.py)

</details>

## Evidence

- [STAR_TRANSPORT_FIDELITY_PROBE_v2.json#L1](STAR_TRANSPORT_FIDELITY_PROBE_v2.json#L1)
- [F3A_K22_CARRIER_BILLING.json#L1](F3A_K22_CARRIER_BILLING.json#L1)
- [AB_K22R_PROBE_v2.json#L1](AB_K22R_PROBE_v2.json#L1)
- [PILOT_RAW_RESULT.json#L1](PILOT_RAW_RESULT.json#L1)
- [REFIT_DIAGNOSTIC.json#L1](../carrier-host-refit-ceiling-measured-rejected/REFIT_DIAGNOSTIC.json#L1)
- [AUDIT_RECEIPT.json#L1](AUDIT_RECEIPT.json#L1)
- [AUDIT_RECEIPT_v2.json#L1](AUDIT_RECEIPT_v2.json#L1)

<details>
<summary>All result files and records (7)</summary>

- [STAR_TRANSPORT_FIDELITY_PROBE_v2.json](STAR_TRANSPORT_FIDELITY_PROBE_v2.json) · result
- [F3A_K22_CARRIER_BILLING.json](F3A_K22_CARRIER_BILLING.json) · result
- [AB_K22R_PROBE_v2.json](AB_K22R_PROBE_v2.json) · result
- [PILOT_RAW_RESULT.json](PILOT_RAW_RESULT.json) · result
- [REFIT_DIAGNOSTIC.json](../carrier-host-refit-ceiling-measured-rejected/REFIT_DIAGNOSTIC.json) · result
- [AUDIT_RECEIPT.json](AUDIT_RECEIPT.json) · result
- [AUDIT_RECEIPT_v2.json](AUDIT_RECEIPT_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-rank-4-fourth-cumulant-two-pair-residual-carrier-host) · [Copy provenance](../../../PROVENANCE.json)

