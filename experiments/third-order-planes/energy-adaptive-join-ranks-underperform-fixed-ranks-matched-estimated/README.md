# Energy-adaptive join ranks underperform fixed ranks at matched estimated work

2026-09-25T08:35Z · experiment · rejected · measured

Captured energy varies too little across networks and poorly predicts tolerance to rank loss. The only threshold making network-dependent choices was worse than the fixed-rank tradeoff.

Recorded measurement:

    - 2026-09-25 01:47 analyst: ENERGY-ADAPTIVE JOIN RANK CLOSED (fable_analyst_v1/RESULT.md section 39; adaptive_join.py on the S23-structure torch card s6bcard_loop4j, loopR1 + round-1 dials, VALID16, CPU gated; energy-ordered Nystrom join so every truncation's captured energy is free). At matched estimated bill (porter unit costs x r/384): adaptive tau .99976 (the only tau with per-net choices, at L10) = raw +2.06% at bill -3.18% = +0.28 pt WORSE than the fixed-rank curve and +0.40 pt worse than the fixed schedule 384/352/320 (+1.66%). The other taus reproduce fixed per-join schedules exactly. Missed energy barely varies across nets (+-10-20%) and does not predict the per-net penalty (corr -0.58..+0.18). Fixed ranks vs 384: 352 +1.25% raw (est. bill -2.67%), 320 +4.04% (-5.35%). A 320/384 50/50 mix is worse than 352 (convex). Per-net penalties at 320 span -0.7..+8.6%, so a 320/384 oracle is worth -0.3..-0.8 pt adjusted, but not with this signal. Side note: uniform 320 at -0.5..-1.6% adjusted depends only on the native bill; a porter census of a tree-sized rank-320 S23 would settle it.
    - 2026-09-25 01:35 lead: ENERGY-ADAPTIVE JOIN RANK CLOSED (analyst §39, torch VALID16, S23 structure). Missed join energy barely varies across nets and doesn't predict tolerance (corr -0.58..+0.18). The only per-net-choosing threshold is 0.28 pt worse than the fixed-rank curve; the fixed front-loaded schedule (384/352/320) is 0.12 pt under it (noise).
      Fixed ranks vs 384: 352 +1.25% raw at est. bill -2.67%; 320 +4.04% raw at est. bill -5.35%; a 320/384 mix is worse than 352 (raw is convex in rank).
      Open: uniform rank 320 is a small cost lever ONLY if the native bill falls >= ~4.5%. Porter to run a native census of S23@R320 after S24.
    - 2026-09-25 01:36 analyst: TIMESTAMP CORRECTION for my own lines above (content unchanged): "01:20 analyst: k4 CARRIER" was written at ~01:15, "02:00 analyst: PER-NET ADAPTIVE ALLOCATION" at ~01:17, "01:47 analyst: ENERGY-ADAPTIVE JOIN RANK" at ~01:34. I had stamped estimated times; from now on I stamp from `date`.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_analyst_v1/adaptive_join.py](code/fable_analyst_v1/adaptive_join.py)
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
<summary>Additional programs and dependencies (23)</summary>

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
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop4j.py](code/fable_physicist_v1/refit6b/s6bcard_loop4j.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L879](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L879)
- [TEAM_LOG_2026-09-22_to_09-26.md#L880](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L880)
- [TEAM_LOG_2026-09-22_to_09-26.md#L881](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L881)
- [TEAM_LOG_2026-09-22_to_09-26.md#L882](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L882)
- [TEAM_LOG_2026-09-22_to_09-26.md#L883](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L883)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT-1a40d4c3.md](../gain-mixture-correction-half-strength-native-adjusted-ratio-1/RESULT-1a40d4c3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-energy-adaptive-join-ranks-underperform-fixed-ranks-matched-estimated) · [Copy provenance](../../../PROVENANCE.json)

