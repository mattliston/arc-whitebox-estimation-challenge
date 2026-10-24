# Network-adaptive rank-16 cross-unit residual covariance observer: measured and rejected

2026-08-25 · experiment · rejected · measured

The independently verified fourfold/inner-cross-fit rank-16 covariance observer is finite on all 688 out-of-fold (OOF) rows at final mean squared error (MSE) 3.230119894372402e-8 versus J 3.2334958229470984e-8, only a 0.1044049% improvement. Fold MSEs are 3.151103526735653e-8, 3.267479855636035e-8, 3.3542892093804806e-8, and 3.1490923178010657e-8; maximum per-network MSE is 3.4171608557348345e-7 at rank643/mlp364. It fails the aggregate, every-fold, and maximum-network accuracy gates; observer_selection remained sealed and no name join occurred.

Recorded measurement:

    The independently verified fourfold/inner-cross-fit rank-16 covariance observer is finite on all 688 OOF rows at final MSE 3.230119894372402e-8 versus J 3.2334958229470984e-8, only a 0.1044049% improvement. Fold MSEs are 3.151103526735653e-8, 3.267479855636035e-8, 3.3542892093804806e-8, and 3.1490923178010657e-8; maximum per-network MSE is 3.4171608557348345e-7 at rank643/mlp364. It fails the aggregate, every-fold, and maximum-network accuracy gates; observer_selection remained sealed and no name join occurred. Read-only decomposition shows 86.694% cancellation and alignment—not covariance amplitude—explains gain. Even exact truth-derived coefficients in its one dynamic rank-16 span reach only 2.8928320722187982e-8, recovering 10.535% of J endpoint error versus 69.076% required. CV, verification, and decomposition are frozen at SHA-256 37187da1fabdc84951a5bc68ef3f12fba1d4b940bc19027c8426f59f1b61be37, 9b6a00d819eb219c8eada6242d96821a76abbfda2076e21f263d079113cb7645, and 007ba020154fed4ca2dd63f16fd10b4961c85f4bd8f5b719cc131acafb53f82a.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_backward_adjoint_state_cv_v1/analyze_m_vs_j.py](code/public_full688_backward_adjoint_state_cv_v1/analyze_m_vs_j.py)
- [code/public_full688_rank16_cross_unit_covariance_cv_v1/run_experiment.py](code/public_full688_rank16_cross_unit_covariance_cv_v1/run_experiment.py)
- [code/public_full688_rank16_cross_unit_covariance_cv_v1/analyze_n_vs_j.py](code/public_full688_rank16_cross_unit_covariance_cv_v1/analyze_n_vs_j.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_local_innovation_transport_v1/run_experiment.py](code/public_full688_local_innovation_transport_v1/run_experiment.py)
- [code/public_full688_rank16_cross_unit_covariance_cv_v1/verify_results.py](code/public_full688_rank16_cross_unit_covariance_cv_v1/verify_results.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)

## Evidence

- [CV_RESULTS.json#L1](../learned-backward-adjoint-residual-observer-rejected-full-width-cross/CV_RESULTS.json#L1)
- [VERIFICATION.json#L1](../learned-backward-adjoint-residual-observer-rejected-full-width-cross/VERIFICATION.json#L1)
- [M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json#L1](../learned-backward-adjoint-residual-observer-rejected-full-width-cross/M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json#L1)
- [PREREGISTRATION.json#L7](PREREGISTRATION.json#L7)
- [RUN_MANIFEST.json#L1](RUN_MANIFEST.json#L1)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [STATIC_RECEIPT.json#L1](STATIC_RECEIPT.json#L1)
- [SMOKE_RECEIPT.json#L1](SMOKE_RECEIPT.json#L1)
- [TRAINING_RELEASE.json#L1](TRAINING_RELEASE.json#L1)
- [CV_RESULTS.json#L1](CV_RESULTS.json#L1)
- [VERIFICATION.json#L1](VERIFICATION.json#L1)
- [N_VS_J_OOF_DECOMPOSITION_AND_ONE_GLOBAL_ORACLE.json#L142](N_VS_J_OOF_DECOMPOSITION_AND_ONE_GLOBAL_ORACLE.json#L142)

<details>
<summary>All result files and records (12)</summary>

- [CV_RESULTS.json](../learned-backward-adjoint-residual-observer-rejected-full-width-cross/CV_RESULTS.json) · result
- [VERIFICATION.json](../learned-backward-adjoint-residual-observer-rejected-full-width-cross/VERIFICATION.json) · result
- [M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json](../learned-backward-adjoint-residual-observer-rejected-full-width-cross/M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json) · result
- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_MANIFEST.json](RUN_MANIFEST.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [STATIC_RECEIPT.json](STATIC_RECEIPT.json) · result
- [SMOKE_RECEIPT.json](SMOKE_RECEIPT.json) · result
- [TRAINING_RELEASE.json](TRAINING_RELEASE.json) · result
- [CV_RESULTS.json](CV_RESULTS.json) · result
- [VERIFICATION.json](VERIFICATION.json) · result
- [N_VS_J_OOF_DECOMPOSITION_AND_ONE_GLOBAL_ORACLE.json](N_VS_J_OOF_DECOMPOSITION_AND_ONE_GLOBAL_ORACLE.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-network-adaptive-rank-16-cross-unit-residual-covariance-observer) · [Copy provenance](../../../PROVENANCE.json)

