# Learned backward-adjoint residual observer: rejected in full-width cross-validation

2026-08-24 · experiment · rejected · measured

The independently verified fourfold full-688 backward-adjoint result is finite at out-of-fold (OOF) final mean squared error (MSE) 3.233766009119748e-8, a 0.008355853% regression versus J's 3.2334958229470984e-8. Its maximum per-network MSE is 3.3816765796841085e-7, so both the frozen mean and maximum gates fail. Fold direction splits 2–2, 98.699% of paired movement cancels, and all four >1e-7 breaches survive; this free backward feature path does not impose a globally consistent residual smoother.

Recorded measurement:

    The independently verified fourfold full-688 backward-adjoint result is finite at OOF final MSE 3.233766009119748e-8, a 0.008355853% regression versus J's 3.2334958229470984e-8. Its maximum per-network MSE is 3.3816765796841085e-7, so both the frozen mean and maximum gates fail. Fold direction splits 2–2, 98.699% of paired movement cancels, and all four >1e-7 breaches survive; this free backward feature path does not impose a globally consistent residual smoother. The exact result, verification, and one-successor decomposition are bound at SHA-256 1c8fc334b1ba20fba82a82ced63361c9ff42ad079bb9a7ec54a5b1c9d7e2ce7b, a69e5f5b417486ba3f55583c6db096bca2233798df7064e0fcad1587d472f071, and 8ee24497e1bd763ec1f4c001af4a0ae74b558c7ee888fddf351f2ad0249b7fdb.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_bidirectional_adjoint_smoother_design_v1/reproduce_readonly_diagnostic.py](code/public_full688_bidirectional_adjoint_smoother_design_v1/reproduce_readonly_diagnostic.py)
- [code/public_full688_backward_adjoint_state_cv_v1/run_experiment.py](code/public_full688_backward_adjoint_state_cv_v1/run_experiment.py)
- [code/public_full688_backward_adjoint_state_cv_v1/analyze_m_vs_j.py](code/public_full688_backward_adjoint_state_cv_v1/analyze_m_vs_j.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)

## Evidence

- [DIAGNOSTIC_RESULT.json#L1](DIAGNOSTIC_RESULT.json#L1)
- [DESIGN_RECEIPT.json#L8](DESIGN_RECEIPT.json#L8)
- [PREREGISTRATION.json#L7](PREREGISTRATION.json#L7)
- [RUN_MANIFEST.json#L1](RUN_MANIFEST.json#L1)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [SMOKE_RECEIPT.json#L1](SMOKE_RECEIPT.json#L1)
- [TRAINING_RELEASE.json#L1](TRAINING_RELEASE.json#L1)
- [CV_RESULTS.json#L1](CV_RESULTS.json#L1)
- [VERIFICATION.json#L67](VERIFICATION.json#L67)
- [M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json#L1011](M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json#L1011)

<details>
<summary>All result files and records (10)</summary>

- [DIAGNOSTIC_RESULT.json](DIAGNOSTIC_RESULT.json) · result
- [DESIGN_RECEIPT.json](DESIGN_RECEIPT.json) · result
- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_MANIFEST.json](RUN_MANIFEST.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [SMOKE_RECEIPT.json](SMOKE_RECEIPT.json) · result
- [TRAINING_RELEASE.json](TRAINING_RELEASE.json) · result
- [CV_RESULTS.json](CV_RESULTS.json) · result
- [VERIFICATION.json](VERIFICATION.json) · result
- [M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json](M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-learned-backward-adjoint-residual-observer-rejected-full-width-cross) · [Copy provenance](../../../PROVENANCE.json)

