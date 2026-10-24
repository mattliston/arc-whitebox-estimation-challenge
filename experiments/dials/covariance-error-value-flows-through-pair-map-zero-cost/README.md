# Covariance error value flows through the pair map; zero-cost Gauss-Newton knobs improve held-out error

2026-09-23T05:29Z · experiment · adopted · measured

Gauss-Newton fitting adjusted the kink and non-kink (2,1) rows, (3,1) covariance scale, message edge and recentering. The native implementation uses a closed-form covariance delta rather than a second pair-map evaluation.

Recorded measurement:

    2026-09-22 22:29 covhunter: VALUE FLOW + knob screen. The first-order value of the exact carried C flows 100% through the PAIR MAP's use of Cpre (adjoint with one leaf per consumer): post covariance -> next layers 50-90%, per-unit variance -> post mean 20-50%; post plane/star/births/radial/K22/features/edge ~0. Injection structure: K21-kink and K31 truth corrections are orthogonal in the covariance (cos -.02..-.10), not row scalings (<=8%/18% energy), broad per-pair. Gauss-Newton screen of zero-cost covariance-channel knobs on top of the K31 row factor (.85/.5 L1-14; fitted on TRAIN 35/100/40/41/43/44, quadratic model accurate to 0.5%): iso -0.091, k21n -0.029, k31(cov) -0.259, k22 -0.062, edge +0.32 -> VALID40 torch vs row factor 0.9869 (CI .9806-.9934) 30/40 wins, worst +3.9%; vs bank 0.9728 36/40 (row alone 0.9858 40/40, reproduces physicist). Wider fit (18 TRAIN nets, + early edge L4-7 knob) running.
    2026-09-22 23:42 covhunter: DONE (fable_covhunter_v1/RESULT.md). Zero-bill covariance-channel knob set on top of the K31 row factor (.85/.5 L1-14), Gauss-Newton fit on 18 TRAIN nets: kink-row K21 += -0.094 k3 C/v, non-kink K21 x0.972, T31 x0.788 (all three in the covariance output only, L2-14), message edge x1.164, recentering x0.92 -> VALID40 torch vs row factor 0.9834 (CI .9789-.9883) 32/40 wins, worst +2.3%; vs bank 0.9694 (38/40). Native form = closed-form delta on the pair map's covariance output (certified vs second call to 1e-4, cert_delta.py), ~30 n^2 ops/layer, no second pair call. Learned-OFF check: truth K21 kink rows still overshoots at eps 1 (+10%/+6%), K31 truth -20%: the overshoot is physics-level, not co-adaptation.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_covhunter_v1/cert_delta.py](code/fable_covhunter_v1/cert_delta.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (12)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/gn2_knobs.py](code/fable_covhunter_v1/gn2_knobs.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L67](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L67)
- [TEAM_LOG_2026-09-22_to_09-26.md#L94](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L94)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../../../records/research/2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/RESULT.md) · record

</details>

## Related work

- shipped in: [Submission #331961: held views, copy elimination, large arenas, and fitted covariance scales](../../../submissions/331961/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-covariance-error-value-flows-through-pair-map-zero-cost) · [Copy provenance](../../../PROVENANCE.json)

