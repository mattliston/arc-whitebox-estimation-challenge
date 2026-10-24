# Joint cubic shell and fourth-order carrier: corrected error 3.1339e-8 versus 3.1334e-8 baseline

2026-08-25 · experiment · rejected · measured

the joint shell+carrier host's refit floor is FLAT - 3.1339e-8 versus the base host's 3.1334e-8 (+0.014 percent) under the frozen 160/40 protocol - while the HOST raw keeps compounding: 9.04e-8 base -> 5.83e-8 carrier -> 5.03e-8 joint on the valid-40 rows (minus 44 percent total from analytic fixes). Verdict: analytic kappa-family removers compose on the host but remove only tape-reconstructable error; the corrected endpoint and therefore every corrector-based deployment of this family stays at about 3.13e-8 dev out-of-sample. The composed card + retrained corrector cannot beat the 2.2843e-8 bank (parity raw at utilization 0.889 versus 0.78) - fire OFF; A8A completes as the deployed-frame confirmation receipt.

Recorded measurement:

    CLOSED MEASURED: the joint shell+carrier host's refit floor is FLAT - 3.1339e-8 versus the base host's 3.1334e-8 (+0.014 percent) under the frozen 160/40 protocol - while the HOST raw keeps compounding: 9.04e-8 base -> 5.83e-8 carrier -> 5.03e-8 joint on the valid-40 rows (minus 44 percent total from analytic fixes). Verdict: analytic kappa-family removers compose on the host but remove only tape-reconstructable error; the corrected endpoint and therefore every corrector-based deployment of this family stays at about 3.13e-8 dev out-of-sample. The composed card + retrained corrector cannot beat the 2.2843e-8 bank (parity raw at utilization 0.889 versus 0.78) - fire OFF; A8A completes as the deployed-frame confirmation receipt. The live raw lane is unchanged: state fidelity (A6) - the propagated cumulant state is measurably undersized and the machinery is 8.2e-9-class on true state.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_joint_floor_probe_v1/estimator_joint.py](code/public_joint_floor_probe_v1/estimator_joint.py)
- [code/public_joint_floor_probe_v1/generate_joint_rows.py](code/public_joint_floor_probe_v1/generate_joint_rows.py)
- [code/public_joint_floor_probe_v1/refit_joint.py](code/public_joint_floor_probe_v1/refit_joint.py)
- [code/public_joint_floor_probe_v1/run_joint_generation.py](code/public_joint_floor_probe_v1/run_joint_generation.py)

## Evidence

- [JOINT_FLOOR_RESULT.json#L1](JOINT_FLOOR_RESULT.json#L1)

<details>
<summary>All result files and records (1)</summary>

- [JOINT_FLOOR_RESULT.json](JOINT_FLOOR_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-joint-cubic-shell-fourth-order-carrier-corrected-error-3) · [Copy provenance](../../../PROVENANCE.json)

