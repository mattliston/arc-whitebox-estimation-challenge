# Full-rank nonlinear weight-space round-trip residual operator: measured and rejected

2026-08-25 · experiment · rejected · measured

The single sealed, independently audited full-rank nonlinear W/W^T round-trip arm is terminal. Verified out-of-fold (OOF) final mean squared error (MSE) is 3.233151267667713e-8 versus frozen J 3.2334958229470984e-8, an improvement of only 0.0106558%; all four folds improve by 0.0057% to 0.0179%, wins/losses are 439/249, and 58.241% of gross paired row movement cancels. The maximum per-network MSE 3.409399173476085e-7 at rank 643 preserves the inherited hard breach.

Recorded measurement:

    The single sealed, independently audited full-rank nonlinear W/W^T round-trip arm is terminal. Verified OOF final MSE is 3.233151267667713e-8 versus frozen J 3.2334958229470984e-8, an improvement of only 0.0106558%; all four folds improve by 0.0057% to 0.0179%, wins/losses are 439/249, and 58.241% of gross paired row movement cancels. The maximum per-network MSE 3.409399173476085e-7 at rank 643 preserves the inherited hard breach. The gate fails every accuracy conjunct. The truth-optimal emitted-direction oracle caps at 3.216608956319193e-8, capturing 0.5222% of J endpoint error versus 69.076% needed: the learned full-rank direction cannot even point at the residual error. Independent replay verification passed with the frozen-J positive control matching J's CV records bit-exactly on all four folds. Combined with M (future-raw adjoint context, -0.008%) and N (rank-16 covariance, +0.104%, span oracle 10.535%), this closes the observer-on-fixed-tape family as information-capped near 3.23e-8 while teacher forcing at 8.207963477637236e-9 proves the recursion has capacity when the true state is supplied.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_fullrank_roundtrip_residual_cv_v1/analyze_p_vs_j.py](code/public_full688_fullrank_roundtrip_residual_cv_v1/analyze_p_vs_j.py)
- [code/public_full688_fullrank_roundtrip_residual_cv_v1/run_experiment.py](code/public_full688_fullrank_roundtrip_residual_cv_v1/run_experiment.py)
- [code/public_full688_fullrank_roundtrip_residual_cv_v1/verify_results.py](code/public_full688_fullrank_roundtrip_residual_cv_v1/verify_results.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_local_innovation_transport_v1/run_experiment.py](code/public_full688_local_innovation_transport_v1/run_experiment.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)

## Evidence

- [PREREGISTRATION.json#L6](PREREGISTRATION.json#L6)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [AUDIT_RECEIPT.json#L1](AUDIT_RECEIPT.json#L1)
- [TRAINING_RELEASE.json#L1](TRAINING_RELEASE.json#L1)
- [CV_RESULTS.json#L1](CV_RESULTS.json#L1)
- [VERIFICATION.json#L1](VERIFICATION.json#L1)
- [P_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json#L1](P_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json#L1)
- [ONE_GRAPH_READY_SUCCESSOR_PROPOSAL.json#L13](ONE_GRAPH_READY_SUCCESSOR_PROPOSAL.json#L13)

<details>
<summary>All result files and records (8)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [AUDIT_RECEIPT.json](AUDIT_RECEIPT.json) · result
- [TRAINING_RELEASE.json](TRAINING_RELEASE.json) · result
- [CV_RESULTS.json](CV_RESULTS.json) · result
- [VERIFICATION.json](VERIFICATION.json) · result
- [P_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json](P_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json) · result
- [ONE_GRAPH_READY_SUCCESSOR_PROPOSAL.json](ONE_GRAPH_READY_SUCCESSOR_PROPOSAL.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-full-rank-nonlinear-weight-space-round-trip-residual-operator) · [Copy provenance](../../../PROVENANCE.json)

