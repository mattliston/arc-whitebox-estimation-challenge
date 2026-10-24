# Nyström residual-plane range finder with one power iteration improves native adjusted error 0.320%

2026-09-25T07:07Z · experiment · adopted · measured

Selecting high-norm columns without a power step failed, but retaining the power step passed the native forty-network keep gate. It was stacked with the next substantive change.

Recorded measurement:

    2026-09-25 00:07 porter: STAR NYSTROM RANGE FINDER (lead 23:5x; fable_porter_v1/star_nys_test.py + RESULT_star_nys_PT48.txt; torch VALID16, S23-structure card, loopR1 + round-1 dials, no refit; control certified equal to lrstar_s1zero_hook on net 37). nys (48 largest-norm columns, no power step): raw +33.8% [CI +27.3, +41.2], 0/16 better, native bill -1.64% -> DEAD (kept rank-32 energy .755 vs .804: the column span misses the directions the card uses). nysp (same columns + the power step): raw +0.083% [CI -0.15, +0.31], 4/16 better, native bill -0.44% (only A Omega goes; the power step's two products stay), adjusted est. -0.36% -> PASSES the raw <= +0.3% bar. Native in-process 37/39: MSE +0.36%/+0.10% (torch +0.32/+0.05). Native VALID40 of S23 + nysp vs the S23 rows RUNNING as a measurement only (valid40_S23_lrnys_q1.out; stack/S23_lrnys_q1_TEST.py = S23 fire copy + stack/lrnys_fragment_q1.py; einsum PASS, validate PASS); no staging.
    - 2026-09-25 00:08 lead: STAR NYSTROM RANGE FINDER, final (porter; torch VALID16, S23 structure): without the power step raw +33.8% (dead); WITH the power step (48 top-norm columns replace Gaussian Omega) raw +0.083% [CI -0.15, +0.31], 4/16, native bill -0.44% (only A Omega goes; the power step's two products stay), adjusted ~-0.36%. Native 37/39 +0.36/+0.10% (torch +0.32/+0.05%). Native VALID40 running as a measurement (stack/S23_lrnys_q1_TEST.py). A stacking lever, not a fire: it rides with the next real change if VALID40 shows adj <= -0.25% and raw <= +0.3%.
    2026-09-25 00:19 porter: STAR NYSTROM (power step) native VALID40 vs the S23 rows: adjusted -0.320% [-0.488, -0.156] 30/40, raw +0.137% [-0.035, +0.309] 18/40, util -0.455%; 0 errors, all finite. The lead's keep bar (adj <= -0.25%, raw <= +0.3%) is MET: it stays in the stack as stack/lrnys_fragment_q1.py (appended after S23) and rides with the next real change. JOIN SCHEDULE native bills (S23 structure, prebuild_S23struct_sched.sh; every3 = S23 byte-exact): even vs every3 bill -0.07% (r35) / -0.47% (r100), native MSE -1.06% / +0.25%; covhunter's 4-join pick next.
    - 2026-09-25 00:19 lead: STAR-NYSTROM KEPT IN STACK (porter native VALID40 vs S23 rows): adjusted -0.320% [CI -0.488, -0.156], 30/40; raw +0.137% [CI -0.035, +0.309]; util -0.455%. stack/lrnys_fragment_q1.py rides with the next real change: E-71 group 1, joint refit loopR3, or a depth-7/routing move. The residual gate runs on the composition. The join-schedule line is killed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_porter_v1/stack/S23_lrnys_q1_TEST.py](code/fable_porter_v1/stack/S23_lrnys_q1_TEST.py)
- [code/fable_porter_v1/stack/lrnys_fragment_q1.py](code/fable_porter_v1/stack/lrnys_fragment_q1.py)
- [code/fable_porter_v1/prebuild_S23struct_sched.sh](code/fable_porter_v1/prebuild_S23struct_sched.sh)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)

<details>
<summary>Additional programs and dependencies (24)</summary>

- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L809](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L809)
- [TEAM_LOG_2026-09-22_to_09-26.md#L810](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L810)
- [TEAM_LOG_2026-09-22_to_09-26.md#L821](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L821)
- [TEAM_LOG_2026-09-22_to_09-26.md#L822](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L822)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT_star_nys_PT48.txt](RESULT_star_nys_PT48.txt) · record

</details>

## Related work

- shipped in: [Submission #332255: joint refit, arithmetic-level fusion, and a Nyström residual-plane range finder](../../../submissions/332255/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-nystrom-residual-plane-range-finder-one-power-iteration-improves) · [Copy provenance](../../../PROVENANCE.json)

