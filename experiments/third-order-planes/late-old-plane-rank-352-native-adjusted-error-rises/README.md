# Late old-plane rank 352: native adjusted error rises 0.12% and 0.90%

2026-09-23T07:15Z · experiment · rejected · measured

Reducing the late frame rank saved much less native work than dense arithmetic estimates predicted because the multiplication kernel changed shape. Read-scale compensation did not recover the raw-error loss.

Recorded measurement:

    2026-09-23 00:15 analyst: S2 native bill by layer (r35, fable_analyst_v1/bill_by_layer.py): L9-14 = 47.3% (newborn push 9.9, slr_reconstruct 7.9, old384 read 7.2, slr_inner 5.5, star 5.3, old384 push 3.9, covariance 3.5), L13-15 = 20.1%; tier costs that scale with the frame rank at L10-15 ~21% of the bill.
    2026-09-23 00:15 analyst: DEEP DIAL (a) tier frame rank at the every3 joins L10/L13 (S2 torch card, learned ON, no refit; fable_analyst_v1/deep_dials.py), final MSE r35/r100/r37/r39: 352 +0.60/+1.52/+1.37/-0.10 (geo +0.84%, bill est -1.8%); 320 +0.63/+5.19/+5.06/-0.25 (geo +2.63%, est -3.5%); 384@10+320@13 geo +1.37% (est -1.8%); 288 r35 +8.2% (est -5.3%), 256 r35 +13.9% (est -7%). Compensation (old-tier read x(1+d), fitted on TRAIN 40/41): best d = 0 (d=+-0.05 worse on 40/41); 288 + d=.1/.2 still +4-23%. Fails the bar.
    2026-09-23 00:36 analyst: NATIVE tier rank 352 at the L10/L13 joins (fable_join_v1/build_join_schedule_v2_rank.py, versioned from build_join_schedule_v1_fired_S2.py: per-join rank + _slr rank checks + envelope width; plain every3 = byte-identical to stack_S3.py, every3@384,384,384 = identical MSE and flops). On S3 (stack_S3 pre-join intermediate): r35 util .18411 -> .18337 (-0.40%), MSE +0.52% -> adjusted +0.12%; r100 util .18179 -> .18070 (-0.60%), MSE +1.52% -> adjusted +0.90%. Fails the <= -0.5% bar; no porter hand-off, no VALID40. Why: the fast-multiply kernel layer eats the saving (r35 per-function: _bh_leaf +1.77%, _bh_backward -0.98%, _bh_forward -0.56%, old384 push -0.36%, slr_inner -0.33%): 352 = 11 x 32 splits worse than 384 = 3 x 128, so leaf work grows.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_analyst_v1/bill_by_layer.py](code/fable_analyst_v1/bill_by_layer.py)
- [code/fable_analyst_v1/deep_dials.py](code/fable_analyst_v1/deep_dials.py)
- [code/fable_join_v1/build_join_schedule_v2_rank.py](code/fable_join_v1/build_join_schedule_v2_rank.py)
- [code/fable_join_v1/build_join_schedule_v1_fired_S2.py](code/fable_join_v1/build_join_schedule_v1_fired_S2.py)
- [code/fable_porter_v1/stack/stack_S3.py](code/fable_porter_v1/stack/stack_S3.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)

<details>
<summary>Additional programs and dependencies (15)</summary>

- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
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
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L112](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L112)
- [TEAM_LOG_2026-09-22_to_09-26.md#L113](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L113)
- [TEAM_LOG_2026-09-22_to_09-26.md#L124](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L124)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-late-old-plane-rank-352-native-adjusted-error-rises) · [Copy provenance](../../../PROVENANCE.json)

