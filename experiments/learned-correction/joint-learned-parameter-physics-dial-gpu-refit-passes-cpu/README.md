# Joint learned-parameter and physics-dial GPU refit passes: CPU validation improves 0.525%

2026-09-25T07:27Z · experiment · adopted · measured

Trainable dial tensors were certified against the native-shared implementation. A joint epoch passed the small improvement gate on both GPU and CPU, with little coefficient drift.

Recorded measurement:

    - 2026-09-25 00:27 [physicist] loopR3 (JOINT refit, learned parts + dials) BUILT AND CERTIFIED; PRE-REGISTERED BAR: VALID16 <= 0.995 x its own valid_before.
      Card refit6b/s6bcard_loop4j.py = covhunter's s6bcard_joint2t (dials as tensors) + join_nystrom_torch + lrstar s1=0 + gn_linear_hook + pair_series, in covhunter's round-3b hook order; fine_ckpt=False + pair_series loop form. Driver refit6b/refit_mps_v3j.py = versioned copy of the engineer's refit_mps_v3.py + one Adam group per trained dial (same warmup/cosine), separate dial clip, dials logged per update, dials_epoch<N>.json saved.
      Certificate (net 37, loopR1 params, round-1 dials): CPU forward BIT-EXACT vs covhunter's round-3b S23 build (max diff 0.0; MSE 1.67403e-8); MPS MSE +0.013% vs CPU; MPS dial gradients within 1.2% of CPU float64 (s_carry -0.02%), learned-group grad norm -0.08%. MPS fix: dials are float64 CPU leaves, the MPS card gets float32 CPU views (autograd casts a grad to the leaf dtype before moving a 0-dim grad off MPS). 2-update smoke OK (~18 s/update; epoch ~15 min).
      Run: from loopR1 params + dials_r1_ep2.json, 1 epoch full+P (lr 1e-4 / 5e-5), X3 and c1 frozen. Waiting only on covhunter's per-dial LR table; fallback = LR_k ~ GN step_k / 20 (s_carry 2.5e-4, s_new/bscale 1e-3, X1/c0 2.5e-3, X0/X2 4e-3, k4a-c 5e-3).
    - 2026-09-25 00:59 [physicist] loopR3 (JOINT refit, learned parts + 10 dials, S23-structure card s6bcard_loop4j, MPS float32, fine_ckpt=False + pair_series loop form) PASSES its pre-registered bar BY A HAIR: VALID16 1.65645e-8 -> 1.64757e-8 = -0.536% pooled (bar -0.5% = 1.64817e-8), 13/16 better, median -0.52%, range -1.42..+0.54%. From loopR1 params + round-1 dials, 1 epoch full+P, dial LR = GN step/25 (X3, c1 frozen). Dials moved little: c0 .7089->.6944, X2 .3585->.3516, k4a .9890->.9815, k4b .9626->.9661, k4c .9601->.9528, X1 .5590->.5577, bscale 1.0509->1.0486, s_carry 1.00469->1.00427, s_new .99312->.99395, X0 -.0062->-.0073. Outputs refit6b/s6b_loopR3_joint_s23/ (params_epoch0.pt sha256 0750b69ba263cd9d..., dials_epoch0.json). The 0.036-point margin is inside MPS/CPU noise, so NOT adopted yet: CPU float64 VALID16 of both arms (loopR1+r1 dials vs loopR3+new dials) launched now; then the porter's native VALID40 if it holds. Note: this run shared MPS with other jobs (the refit_mps_v3 lock bug), so its wall time (32 min) is not representative.
    - 2026-09-25 01:07 [physicist] loopR3 CPU float64 CHECK HOLDS (refit6b/refit_mps_v3j2.py --device cpu --eval-only, card s6bcard_loop4j = S23 structure + fine_ckpt=False + pair_series loop form; loopR3_cpucheck_{A,B}/eval_n16_cpu.json): VALID16 start (loopR1 params + round-1 dials) 1.65620e-8 -> end (loopR3 params + loopR3 dials) 1.64751e-8 = -0.525% pooled, 13/16 better, median -0.52%, range -1.41..+0.53%; passes the -0.5% bar on CPU too (thin: 0.025 pt). MPS vs CPU per net <= 0.073% (start), <= 0.036% (end). For the porter if the lead GOes a native VALID40: S23 + loopR3 params (refit6b/s6b_loopR3_joint_s23/params_epoch0.pt, sha256 0750b69ba263cd9d...) + the dial literals in refit6b/s6b_loopR3_joint_s23/dials_epoch0.json (joint2t / native-shared semantics, covhunter's values() export).
    - 2026-09-25 01:06 lead: loopR3 JOINT REFIT PASSES (physicist; learned parts + 10 dials, X3/c1 frozen; S23-structure card, MPS; verified on CPU f64): VALID16 1.65620e-8 -> 1.64751e-8 = -0.525% (13/16; MPS -0.536%). Dials moved little (c0 .709->.694, X2 .358->.352, k4a .989->.981, k4c .960->.953). Params s6b_loopR3_joint_s23/params_epoch0.pt (sha 0750b69b...).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/s6bcard_loop4j.py](code/fable_physicist_v1/refit6b/s6bcard_loop4j.py)
- [code/fable_physicist_v1/refit6b/refit_mps_v3j.py](code/fable_physicist_v1/refit6b/refit_mps_v3j.py)
- [code/fable_engineer_v1/refit_mps_v3.py](code/fable_engineer_v1/refit_mps_v3.py)
- [code/fable_physicist_v1/refit6b/refit_mps_v3j2.py](code/fable_physicist_v1/refit6b/refit_mps_v3j2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)

<details>
<summary>Additional programs and dependencies (30)</summary>

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
- [code/fable_covhunter_v1/s6bcard_jfitR1.py](code/fable_covhunter_v1/s6bcard_jfitR1.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/lrstar_torch_hook.py](code/fable_engineer_v1/lrstar_torch_hook.py)
- [code/fable_engineer_v1/mps_card.py](code/fable_engineer_v1/mps_card.py)
- [code/fable_engineer_v1/mps_lock.py](code/fable_engineer_v1/mps_lock.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop3.py](code/fable_physicist_v1/refit6b/s6bcard_loop3.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L824](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L824)
- [TEAM_LOG_2026-09-22_to_09-26.md#L825](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L825)
- [TEAM_LOG_2026-09-22_to_09-26.md#L826](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L826)
- [TEAM_LOG_2026-09-22_to_09-26.md#L827](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L827)
- [TEAM_LOG_2026-09-22_to_09-26.md#L838](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L838)
- [TEAM_LOG_2026-09-22_to_09-26.md#L839](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L839)
- [TEAM_LOG_2026-09-22_to_09-26.md#L840](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L840)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [dials_r1_ep2.json](dials_r1_ep2.json) · result
- [dials_epoch0.json](dials_epoch0.json) · result

</details>

## Related work

- shipped in: [Submission #332255: joint refit, arithmetic-level fusion, and a Nyström residual-plane range finder](../../../submissions/332255/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-joint-learned-parameter-physics-dial-gpu-refit-passes-cpu) · [Copy provenance](../../../PROVENANCE.json)

