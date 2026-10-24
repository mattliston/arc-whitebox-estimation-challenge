# GPU mixed-precision refit certificate: 466 seconds versus 1454 with matched validation gain

2026-09-25T05:47Z · tool · informational · measured

Float32 GPU training with protected mean products and paired cube sums matched twenty CPU training updates without drift. It passed the speed and validation gates before adoption.

Recorded measurement:

    - 2026-09-24 22:47 [engineer] E-68 FLOAT32 MPS REFIT CARD: ALL THREE STOP RULES PASS (fable_engineer_v1/RESULT.md E-68; mps_card.py, refit_mps_v2.py). (a) pure float32 fails the gradient gate (r39 median 2.1e-2: gradient error ~1e4 x the L15 prediction error via the residual seed); fixed by the step's w64.T@m / w64.T@b in float64 on CPU + the bank k3 sums in +/- PAIR FORM (float32): MSE <= ~0.05%, gradients 1-5e-3. (b) MPS step 0.27-0.29 of the CPU float64 drop-in (5 thr, load 7-9). (c) 20 updates from j05 ep2 (jfitR1 card), same batches: per-update batch loss MPS vs CPU <= 0.021% (no drift), VALID16 (CPU evaluator) 1.651960e-8 vs 1.652000e-8 = -0.0024% (both -0.94% vs start, 13/16); wall 466 s vs 1454 s. E-67 ADDENDUM: at VALID16 the pair series moves the start from 1.666309e-8 (= physicist's loopR1 valid_before, bit-equal) to 1.667714e-8 (+0.084%): 15 nets <= 0.04%, net 42 +1.18% (knife-edge flip); same-card comparisons only.
    - 2026-09-24 22:48 lead: E-68 CERTIFIED (engineer): 20 updates float32 MPS vs float64 CPU from j05 ep2 on the jfitR1 loop card. Train loss <= 0.021% apart per update, no drift. VALID16 after training: MPS -0.945% vs CPU -0.943% (-0.0024% pooled apart, <= 0.049%/net, same 13/16 improved). Wall 466 s vs 1454 s (0.32x): a refit epoch ~16-17 min instead of ~50.
      Adopted from the NEXT refit (loopR2 on CPU continues). The MPS card must carry the s1=0 hook and, once S23 ships, the Nystrom join mirror.
    - 2026-09-24 22:59 [engineer] E-68 on the loopR2 card: refit_mps_v3.py (--card, default s6bcard_loop3 = E-67 drop-in + s1-zero star) certified MPS float32 vs CPU float64 on s6bcard_loop3 (jfitR1 dials): r39 MSE -0.040%, gradients 2.6e-3 median, step 0.32 of CPU; r42 (the knife-edge net) +0.031%, no flip, gradients 1.0e-3, step 0.28. Ready for the physicist's round 2 on the lead's GO; MPS free.

## Programs

- [code/fable_engineer_v1/mps_card.py](code/fable_engineer_v1/mps_card.py)
- [code/fable_engineer_v1/refit_mps_v2.py](code/fable_engineer_v1/refit_mps_v2.py)
- [code/fable_engineer_v1/refit_mps_v3.py](code/fable_engineer_v1/refit_mps_v3.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (27)</summary>

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
- [code/fable_covhunter_v1/s6bcard_jfitR1.py](code/fable_covhunter_v1/s6bcard_jfitR1.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/lrstar_torch_hook.py](code/fable_engineer_v1/lrstar_torch_hook.py)
- [code/fable_engineer_v1/mps_lock.py](code/fable_engineer_v1/mps_lock.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop.py](code/fable_physicist_v1/refit6b/s6bcard_loop.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop3.py](code/fable_physicist_v1/refit6b/s6bcard_loop3.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L723](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L723)
- [TEAM_LOG_2026-09-22_to_09-26.md#L725](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L725)
- [TEAM_LOG_2026-09-22_to_09-26.md#L726](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L726)
- [TEAM_LOG_2026-09-22_to_09-26.md#L731](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L731)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../../experiments/arithmetic-kernels/grouped-winograd-requests-copied-operands-fewer-requests-but-no/RESULT.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-25-gpu-mixed-precision-refit-certificate-466-seconds-versus-1454) · [Copy provenance](../../PROVENANCE.json)

