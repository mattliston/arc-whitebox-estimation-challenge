# Cubic Edgeworth shell: host error fell 9.967% but refitted corrected error rose 0.943%

2026-08-25 · experiment · rejected · measured

The lambda-cubed Edgeworth shell arm is terminal with a dual verdict. Host side: the shell genuinely improves the order-24 Hermite pair-series closure (Q24) host - protected-40 raw host mean squared error (MSE) 8.030968799749751e-8 versus 8.919980524938374e-8, a 9.967% reduction, the campaign's first real host-accuracy improvement, with the weight algebra machine-certified against an independent polynomial construction and every rollout finite at about 2.0e12 FLOPs funded by the conditional repair. System side: the refit-J corrected pilot is 3.136551362360925e-8 versus the baseline 3.107235383605394e-8, a 0.943% regression - J corrected 65.17% of the base host's error but only 60.94% of the smaller X residual, so the shell removes precisely the systematic fourth-order component the observer could already reconstruct.

Recorded measurement:

    The lambda-cubed Edgeworth shell arm is terminal with a dual verdict. Host side: the shell genuinely improves the Q24 host - protected-40 raw host MSE 8.030968799749751e-8 versus 8.919980524938374e-8, a 9.967% reduction, the campaign's first real host-accuracy improvement, with the weight algebra machine-certified against an independent polynomial construction and every rollout finite at about 2.0e12 FLOPs funded by the conditional repair. System side: the refit-J corrected pilot is 3.136551362360925e-8 versus the baseline 3.107235383605394e-8, a 0.943% regression - J corrected 65.17% of the base host's error but only 60.94% of the smaller X residual, so the shell removes precisely the systematic fourth-order component the observer could already reconstruct. The corrected system sits at an invariant near 3.1e-8 (native pilot frame; 3.23e-8 development frame) now measured from BOTH directions: five observer arms on the fixed host and three host arms under the frozen observer form. This is the joint information floor of the current architecture.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_condrepair_deployment_card_v1/estimator.py](code/public_condrepair_deployment_card_v1/estimator.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_lambda3_native_pilot_v1/generate_x_row.py](code/public_lambda3_native_pilot_v1/generate_x_row.py)
- [code/public_lambda3_native_pilot_v1/run_x_pilot.py](code/public_lambda3_native_pilot_v1/run_x_pilot.py)
- [code/public_lambda3_shell_probe_v1/estimator_x.py](code/public_lambda3_shell_probe_v1/estimator_x.py)
- [code/public_lambda3_shell_probe_v1/instrumented_estimator_x.py](code/public_lambda3_shell_probe_v1/instrumented_estimator_x.py)
- [code/public_lambda3_shell_probe_v1/run_lambda3_probe.py](code/public_lambda3_shell_probe_v1/run_lambda3_probe.py)

## Evidence

- [PREREGISTRATION.json#L11](PREREGISTRATION.json#L11)
- [AUDIT_RECEIPT.json#L27](AUDIT_RECEIPT.json#L27)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [PILOT_RESULT.json#L1](PILOT_RESULT.json#L1)
- [X_CLOSURE.json#L7](X_CLOSURE.json#L7)
- [AB_LAMBDA3_PROBE.json#L1](AB_LAMBDA3_PROBE.json#L1)

<details>
<summary>All result files and records (6)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [AUDIT_RECEIPT.json](AUDIT_RECEIPT.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [PILOT_RESULT.json](PILOT_RESULT.json) · result
- [X_CLOSURE.json](X_CLOSURE.json) · result
- [AB_LAMBDA3_PROBE.json](AB_LAMBDA3_PROBE.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-cubic-edgeworth-shell-host-error-fell-9-967-but) · [Copy provenance](../../../PROVENANCE.json)

