# Residual-plane transport cannot be skipped, but exact rank-32 truncation costs only 0.375% raw error

2026-09-25T00:39Z · experiment · adopted · measured

Blockwise zeroing of star transport caused large losses. Exact low-rank truncation preserved most useful information and motivated a native randomized implementation.

Recorded measurement:

    2026-09-24 17:39 porter: LEVERS: star per-layer skip DEAD (+3.6..+22.6% raw); rank-32 star plane NOT a cliff (+0.37% raw, 8 nets, CI incl 0) -> ~-8% bill natively (2 of 4 batched products); pair map DEAD (1.4% of bill). S17-j05 (ae52f49a) chain running (VALID40 vs S16, parity vs torch j05 VALID40, residual vs S16).
    - 2026-09-24 18:25 lead: PORTER LEVER RESULTS (torch s6bcard, B2 ep1, 8 VALID nets, no refit).
      - Star per-block zeroing DEAD (L1-4 +3.6%, L5-8 +22.6%, L9-11 +18.7%, L12-14 +11.9%).
      - STAR-PLANE RANK TRUNCATION IS NOT A CLIFF: rank 32 +0.37% pooled (CI -0.72..+1.56%, 2/8 better), rank 128 +0.49%.
      - Natively, pushed = W^T A = (W^T U)(S V^T) and s2 become O(n^2 r); s1 stays full. That removes 2 of the 4 batched P=2 products (_bb_product = 16.5% of the bill), so ~-8.2% bill, adjusted ~-7.8% before any refit.
      - It is a PLANE carrier (low-rank; the old plane had 98% of its energy at rank 64), not the information-critical young CP content, so the compression closure does not cover it.
      - GO: torch VALID40 at r = 32 and 64, plus r=32 on the j05 card (bar raw <= +0.8%, CI upper <= +1.5%). The engineer drafts the native design (range finder + rank-r push/s2) for S18.
    2026-09-24 17:58 porter: S18 star sizing: gross -8.26% (56 dense products/net, 1.057 GF each), net ~-5.8% bill with a float32 range finder + ONE power iteration (q=0 doubles the propagated star error; q=1 within 3% of exact SVD). Diagonal <1% of plane energy. Torch VALID40 (exact SVD) 12 nets: b2 r32 +0.26%, r64 +0.45%, j05 r32 +0.14%.
    - 2026-09-24 19:10 lead: S18 VALID40 vs S17p PASS (engineer and porter): adjusted -5.35%, 40/40, raw +0.23%. Wall pair user CPU -6.35%, bill -5.60%, expected worst ~69.6 s. Staging after the residual gate (~19:35). Star torch VALID40 final: B2 r32 +.375%, r64 +.290%, j05 r32 +.190% (18/40). RULE: from S19 on, all refits train on the S18 card (low-rank star). The engineer writes a torch hook that matches the native factorization exactly (k=48, 1 power step, rank 32) and certifies it against native. The engineer then runs the covariance diag+low-rank test (r=128/256/512, kill if r256 > +0.5%) and profiles the refit step.

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
<summary>Additional programs and dependencies (20)</summary>

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
- [code/fable_physicist_v1/refit6b/carry/variant_bd.py](code/fable_physicist_v1/refit6b/carry/variant_bd.py)
- [code/fable_physicist_v1/refit6b/s6bcard_j05.py](code/fable_physicist_v1/refit6b/s6bcard_j05.py)
- [code/fable_porter_v1/star_ablation_s6b.py](code/fable_porter_v1/star_ablation_s6b.py)
- [code/fable_porter_v1/star_valid40_report.py](code/fable_porter_v1/star_valid40_report.py)
- [code/fable_porter_v1/star_valid40_s6b.py](code/fable_porter_v1/star_valid40_s6b.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L525](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L525)
- [TEAM_LOG_2026-09-22_to_09-26.md#L526](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L526)
- [TEAM_LOG_2026-09-22_to_09-26.md#L527](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L527)
- [TEAM_LOG_2026-09-22_to_09-26.md#L528](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L528)
- [TEAM_LOG_2026-09-22_to_09-26.md#L529](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L529)
- [TEAM_LOG_2026-09-22_to_09-26.md#L530](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L530)
- [TEAM_LOG_2026-09-22_to_09-26.md#L531](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L531)
- [TEAM_LOG_2026-09-22_to_09-26.md#L534](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L534)
- [TEAM_LOG_2026-09-22_to_09-26.md#L558](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L558)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332226: rank-32 memoryless (2,1) residual plane with one power iteration](../../../submissions/332226/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-residual-plane-transport-cannot-be-skipped-but-exact-rank) · [Copy provenance](../../../PROVENANCE.json)

