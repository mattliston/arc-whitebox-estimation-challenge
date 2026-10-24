# Shared-frame Tucker compression of young covariance legs fails despite 97.6-99.4% retained energy

2026-09-24T18:08Z · experiment · rejected · measured

Projecting young covariance legs into a common frame caused large error increases even when applied only at birth or with learned corrections disabled. Consumer localization showed both the plane read and third-cumulant diagonal needed the missing information.

Recorded measurement:

    - 2026-09-24 11:08 [physicist] CANDIDATE 4 (TUCKER, pinned legs + unit bond exact, covariance legs of ALL young pairs projected onto ONE common top-r frame every layer >= 8; refit6b/tucker_eval.py; torch S6b + B2 ep1; no refit; full-rank control reproduces the card): VALID16 raw ratio vs B2 ep1: r=256 x5.52, r=128 x10.75, r=64 x16.4, all 0/16. Captured weighted covariance-leg energy L8/10/12/14: r=256 .976/.989/.992/.994, r=128 .951/.982/.987/.989, r=64 .890/.967/.977/.982. KILL (r=128 far above +15%). The common frame holds 98-99% of the energy (no Jacobian decorrelation); the card is hypersensitive to the last 1-2% (the old-tier rank-cliff law). Cost check: it would pay only near r=64 (L14 ~7 U vs 10 U; r=128 ~15 U).
    - 2026-09-24 11:15 lead: CANDIDATE 4 CLOSED (physicist tucker_eval.py; exact pinned leg + unit bond, young s-legs in one common top-r frame from L8, no refit, VALID16 vs B2 ep1): r256 x5.52, r128 x10.75, r64 x16.4, 0/16 at every rank. Weighted energy captured is .976-.994 at r256, so one frame DOES hold every cohort's pushed s-legs, and the card is hypersensitive to the last 1-2% (same law as the old-tier rank cliff). Discriminating test now: the same projection with learned parts OFF against a learned-OFF control. A small loss means co-adaptation (a lean card trained from scratch stays live); >=2x means information (the lean route is closed for lossy carriers).
    - 2026-09-24 11:26 [physicist] HYPERSENSITIVITY IS INFORMATION, NOT CO-ADAPTATION (refit6b/tucker_eval_v2.py, VALID16, no refit): (1) Tucker r=256 projecting ONLY each newborn cohort once at birth (learned ON): x4.07 vs B2 ep1 (0/16), so not the per-layer re-projection (all-layers x5.52). (2) Learned parts OFF (GNN head/head_b + edge output/bias zeroed, refit.py --zero-learned convention): control 2.167e-8 (x1.24 vs learned ON); r=256 x4.43, r=128 x8.71 vs that control (0/16 each), against learned ON x5.52/x10.75. The loss is physical: the card's carried covariance needs every young birth's covariance leg to ~99.9%. Any carrier that drops the last 1-2% of young content is closed, with or without retraining.
    - 2026-09-24 11:44 [physicist] CONSUMER LOCALIZATION of the Tucker sensitivity (refit6b/route_eval.py: copy of the HookCard step + a carried projected-minus-exact difference; birth-only r=256 from L9; control reproduces the card; 8 VALID nets vs B2 ep1): all consumers x3.32 (0/8); plane read only x1.79 (0/8); k3 DIAGONAL only x2.64 (0/8); omitted-radial only x1.00; all consumers but first read exact x2.39. r35 diagnostics vs 8M truth: plane change from projection 10.2%/16.2% (L10/L14) vs card own plane error 7.5%/8.5%; k3 change 7.4%/12.8% vs own 5.4%/6.3% (own errors include MC noise). The per-unit k3 diagonal is the most sensitive consumer, but the card's own error there is SMALLER than the projection's: no consumer is shown to be an accuracy lever (the earlier truth-k3 oracle, -3%, agrees). Stopping per the lead.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/tucker_eval.py](code/fable_physicist_v1/refit6b/tucker_eval.py)
- [code/fable_physicist_v1/refit6b/tucker_eval_v2.py](code/fable_physicist_v1/refit6b/tucker_eval_v2.py)
- [code/fable_refit_v1/refit.py](code/fable_refit_v1/refit.py)
- [code/fable_physicist_v1/refit6b/route_eval.py](code/fable_physicist_v1/refit6b/route_eval.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

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
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L340](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L340)
- [TEAM_LOG_2026-09-22_to_09-26.md#L341](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L341)
- [TEAM_LOG_2026-09-22_to_09-26.md#L350](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L350)
- [TEAM_LOG_2026-09-22_to_09-26.md#L362](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L362)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-shared-frame-tucker-compression-young-covariance-legs-fails-despite) · [Copy provenance](../../../PROVENANCE.json)

