# Nested fine checkpoints cost 7-13% training time without reducing peak memory

2026-09-25T03:53Z · tool · informational · measured

Removing nested fine checkpoints preserved gradients exactly while retaining the outer layer checkpoints. The termwise pair series was adopted for subsequent runs.

Recorded measurement:

    - 2026-09-24 20:53 [engineer] LOOP-CARD REFIT STEP (E-67, fable_engineer_v1/RESULT.md): step 24.4 s (r37, 3 thr) = forward 5.7 + layer-checkpoint recompute ~5.6 + nested fine-checkpoint recompute ~2 + gradient math ~11 (flat elementwise). Bottleneck = the pair map: runs 3x per step + backward 3.5x its forward = ~34% of the step. The nested fine checkpoints buy NO memory (peak RSS 8.38 vs 8.39 GB) and cost 7-13%; gradients bit-identical without them. Dropping the per-layer checkpoint needs +11.8..23.8 GB: no. DROP-IN (fine_ckpt=False + pair_series_torch): step x0.820 (r37, jfitON2, 3 thr) and x0.817 (r39, live jfitR1 card, 5 thr); predictions 1e-7, MSE -0.012/-0.003%, gradients median 1.1-1.3e-3 (thread-count floor 1e-4..8e-4; gn_linear_hook level); peak +0.2 GB. Offered to the lead for the refit owners.
    - 2026-09-24 20:59 lead: E-67 GO (engineer): fine_ckpt=False + pair_series loop form, installed from the NEXT runs (covhunter round-2 dial fit, physicist round-2 refit); the running s6b_loopR1 is untouched. Runs must say so in their TAG; comparisons with a margin under 0.1% need both arms on one card. Engineer's next task E-68: float32 refit card on MPS. Stop rules: (a) CPU f32 vs f64 final MSE <= 0.05%, gradient median <= ~1e-2; (b) MPS step <= 1/3 of CPU; (c) 20-update training certificate VALID16 +-0.1%. Kill if f64 islands exceed ~20% of the step's flops or the speedup is under 2x.
    - 2026-09-24 21:02 [physicist] loopR1 valid_before (jfitR1 dials + S18 + gn_linear_hook, ep2 params, VALID16, 5 threads) = 1.666309e-8 vs covhunter's full-recompute 1.666448e-8 (-0.008%: consistent). BAR resolves to VALID16 <= 1.657977e-8. ~86 s/update (was 142 s before gn_linear_hook); VALID16 ~21:45. Round-2 card ready: refit6b/s6bcard_loop2.py = s6bcard_loop + engineer E-67 (nested fine_ckpt off + pair_series loop form); net 39 vs s6bcard_loop: MSE -1.7e-5 rel, max prediction diff 2.5e-7. s6bcard_loop.py left unchanged (the running round-1 card). Round-2 tags will state 'fine_ckpt=False + pair_series loop form'.

## Programs

- [code/fable_physicist_v1/refit6b/s6bcard_loop2.py](code/fable_physicist_v1/refit6b/s6bcard_loop2.py)
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
<summary>Additional programs and dependencies (21)</summary>

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
- [code/fable_covhunter_v1/s6bcard_jfitR1.py](code/fable_covhunter_v1/s6bcard_jfitR1.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/lrstar_torch_hook.py](code/fable_engineer_v1/lrstar_torch_hook.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L621](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L621)
- [TEAM_LOG_2026-09-22_to_09-26.md#L623](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L623)
- [TEAM_LOG_2026-09-22_to_09-26.md#L624](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L624)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../../experiments/arithmetic-kernels/grouped-winograd-requests-copied-operands-fewer-requests-but-no/RESULT.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-25-nested-fine-checkpoints-cost-7-13-training-time-without) · [Copy provenance](../../PROVENANCE.json)

