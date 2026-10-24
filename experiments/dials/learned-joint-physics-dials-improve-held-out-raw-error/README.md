# Learned-on joint physics dials improve held-out raw error 2.23% without a refit

2026-09-25T01:58Z · experiment · adopted · measured

Joint fitting on the deployed learned-on trajectory selected different plane, birth, carried-vector and cumulant strengths than the learned-off fit.

Recorded measurement:

    2026-09-24 18:58 covhunter: LEARNED-ON JOINT GN FIT PASSES (RESULT sec 26): j05 ep0 card, learned ON, same 12 dials: fit in-sample -4.07%, LOO -3.01%, real -4.14%; VALID16 learned ON vs j05 ep0 1.71226e-8: 1.67409e-8 = -2.23% pooled, 13/16 (se 0.58%) -> PURE LITERAL CHANGE, no refit. ON dials (physical units): K31 c0 .85->.703, c1 .5->1.123; k4 x .991/.952/.954 (L1-7/8-11/12-15); j1 .531, j2 +.209, j3 -.117, j0 ~0; b x1.051; carried lam x1.0042/layer, newborn x.9925. (OFF fit: c0 .458, c1 .396, k4 .954/.953/.915, j1 .683, j3 -.415.) Next: re-validate on the S18 hooked card (lrstar) VALID40, after the c0 scan (running).
    - 2026-09-24 19:25 lead: COVHUNTER LEARNED-ON JOINT DIAL FIT (no refit): VALID16 -2.23% vs j05 ep0 (13/16, se .58%); LOO -3.01% on the 8 fit nets.
      Dials (ON vs OFF fit): K31 c0 .703 (.458), c1 1.123 (.396); k4 x.991/.952/.954 (.954/.953/.915); j1 .531 (.683), j2 +.209 (+.295), j3 -.117 (-.415), j0 ~0; b x1.051 (1.026); carried lam x1.0042/layer (1.0076); newborn x.9925 (.987).
      The learned parts absorb most of the extra j1 and the j3 cut, plus ~60% of the c0 cut. Frozen as s6bcard_jfitON1.py (STAR='lr32' option). The c0 scan without j1 already puts the optimum at ~.65-.7 (not .85).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_covhunter_v1/s6bcard_jfitON1.py](code/fable_covhunter_v1/s6bcard_jfitON1.py)
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
<summary>Additional programs and dependencies (18)</summary>

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
- [code/fable_covhunter_v1/s6bcard_joint.py](code/fable_covhunter_v1/s6bcard_joint.py)
- [code/fable_engineer_v1/lrstar_torch_hook.py](code/fable_engineer_v1/lrstar_torch_hook.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L562](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L562)
- [TEAM_LOG_2026-09-22_to_09-26.md#L564](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L564)
- [TEAM_LOG_2026-09-22_to_09-26.md#L565](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L565)
- [TEAM_LOG_2026-09-22_to_09-26.md#L566](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L566)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332227: jointly fitted physics scales with learned correction enabled](../../../submissions/332227/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-learned-joint-physics-dials-improve-held-out-raw-error) · [Copy provenance](../../../PROVENANCE.json)

