# Fourth-cumulant-only oracle gains 4.9%; a source at target accuracy can afford under 3% added work

2026-09-25T07:13Z · experiment · rejected · measured

The combined third- and fourth-cumulant oracle is worth more than either source alone. A practical fourth-only source has too little standalone value to pay for the proposed carriers.

Recorded measurement:

    - 2026-09-25 00:15 [physicist] k4-ALONE PRICE (the cheapest experiment named on the k3/k4 page; RESULT.md section N updated). refit6b/carry/k34_spec_ch.py = versioned copy of the analyst's k34_spec.py with a channel switch (eps/noise ...k3|k4); analyst frame (S6b + B2 ep1, learned ON), 7 held-out nets, 8M truth, L12-14, correction 0.5; deployed and joint rows reproduce the analyst's bit-exactly on all 7 nets. Geo vs deployed: k3+k4 -18.4% (7/7); k3 only -8.7% (7/7); k4 only -4.9% (6/7); k4 only at the spec (truth + 0.25 x card error noise) -3.2% (5/7); k4 only, correction at 1, -2.7% (5/7); control +3.0% (1/7). Joint > sum of parts (-18.4 vs -13.6%). A k4-only source pays only under ~3% of the bill at spec (~5 U), ~5% if exact (~8 U); none reaches 7%. The k4 question is CLOSED. k3 alone is the bigger single lever (-8.7% exact); the restart table (section N.1) says a k3 source must be as good as exact physics from ~L2 to beat the card's bank (exact-from-L5-6 fidelity).
    - 2026-09-25 00:13 lead: k4 QUESTION CLOSED (physicist, analyst's frame, 7 held-out, correction .5): k3+k4 truth -18.4%; k3 alone -8.7%; k4 alone -4.9% exact, -3.2% at the 7% spec, -2.7% with the correction at 1. The joint value exceeds the sum (-18.4 vs -13.6). A k4-only source would have to cost < ~3% of the bill at spec; nothing reaches spec. k3 alone is the bigger single lever, but beating the card's bank (as good as exact physics from L5-6) needs exact-from-L2 fidelity. The only route is the third-order bank's own fidelity, and that needs fourth-order content feeding it (N1.5). CLOSED for sources.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/carry/k34_spec_ch.py](code/fable_physicist_v1/refit6b/carry/k34_spec_ch.py)
- [code/fable_analyst_v1/k34_spec.py](code/fable_analyst_v1/k34_spec.py)
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

<details>
<summary>Additional programs and dependencies (17)</summary>

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

- [TEAM_LOG_2026-09-22_to_09-26.md#L815](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L815)
- [TEAM_LOG_2026-09-22_to_09-26.md#L816](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L816)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-fourth-cumulant-only-oracle-gains-4-9-source-target) · [Copy provenance](../../../PROVENANCE.json)

