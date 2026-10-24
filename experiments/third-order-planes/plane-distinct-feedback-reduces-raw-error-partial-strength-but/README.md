# Plane-to-distinct feedback reduces raw error at partial strength but adjusted error remains worse

2026-09-23T08:02Z · experiment · rejected · measured

Third-cumulant-only feedback was nearly flat. Plane feedback improved raw error at partial strength but its required dense products overwhelmed that gain.

Recorded measurement:

    - 2026-09-23 01:02 analyst: plane-to-distinct FEED (0,1,0) one-step on S2 torch card -> CLOSED. Hook reproduces the 09-02 feed (plane read corr .994-1.000 vs saved read; diag = k3 read 1e-15). Raw geo on 35/100/37/39: k3 L2-14 x1 +1.2% / x.5 -0.5%; k3 L9-14 +0.9/-0.4%; plane L2-14 -0.2/-2.4% (4/4 nets at x.5); plane L9-14 +1.8/-0.6%. Bill: 0.53% per 1024^3 GEMM on S2 -> best adjusted +8.9% (k3 L9-14 x.5), plane L2-14 x.5 +57% (+47% alive/response-restricted). Final-layer k3 read hurts (+0.44% r35). Off-diag read anti-aligned with S2 plane error L4-14 (S2 plane 27-44% too small at L7-14). fable_analyst_v1/RESULT.md sec 20.
    - 2026-09-23 01:06 analyst: feed follow-up (k3-only first, eps 1/.5/.25): k3 read FLAT at every eps (raw geo L2-14 +1.2/-0.5/-0.5%, L9-14 +0.9/-0.4/-0.4%; adjusted >= +8.9%). Plane read overshoots at eps 1 but beats k3-only at .25/.5 (L2-14 -1.8% at .25, -2.4% at .5, 4/4 nets). Lane stays CLOSED on bill. RESULT.md sec 20 addendum.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

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
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)

<details>
<summary>Additional programs and dependencies (15)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_analyst_v1/feed_s2.py](code/fable_analyst_v1/feed_s2.py)
- [code/fable_analyst_v1/feed_s2_diag.py](code/fable_analyst_v1/feed_s2_diag.py)
- [code/fable_analyst_v1/feed_s2_diag2.py](code/fable_analyst_v1/feed_s2_diag2.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/public_g29_bond_kernel_v1/plane_feed_oracle.py](code/public_g29_bond_kernel_v1/plane_feed_oracle.py)
- [code/public_g29_bond_kernel_v1/plane_feed_plane_read.py](code/public_g29_bond_kernel_v1/plane_feed_plane_read.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L130](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L130)
- [TEAM_LOG_2026-09-22_to_09-26.md#L133](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L133)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT-1a40d4c3.md](../gain-mixture-correction-half-strength-native-adjusted-ratio-1/RESULT-1a40d4c3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-plane-distinct-feedback-reduces-raw-error-partial-strength-but) · [Copy provenance](../../../PROVENANCE.json)

