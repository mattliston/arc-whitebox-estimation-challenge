# Cheap fourth-cumulant sources miss the per-unit structure and fail the arithmetic break-even test

2026-09-25T07:03Z · experiment · rejected · measured

Covariance modes, scale mixtures, true-skew laws and free column weights did not reproduce the needed per-unit fourth cumulant. The residual requires hundreds of components and a second full-depth carrier would cost too much.

Recorded measurement:

      (2) Nothing cheap carries the unit-specific 25%, even fitted to truth: covariance-mode shares/alpha/all 19-27%; scale mixture over upstream layer norms (norm_coupling_mc.py) sets the LEVEL (1.2-1.3x) but not the spread (corr .29-.34); truth k3 law 17-21%; free lam4 on the bank's ~21k columns 20-24% held out (col_k4_unit_fit_v2.py; control recovers ~half); rank census of the residual (2,2) plane (k22_rank_census.py, 2 draws): 91% of the per-unit residual needs 256-512 components.
      (3) Economics: +10% bill needs -9.1% raw, +20% needs -16.7%; at 20% even an exact k3+k4 source nets only -2.1%. k4 alone: -4.0% r35 / +9.9% r100 (§34). A second full-depth bank (~80% of the bill) loses even if exact.
      (4) Cheapest experiment that could change it: truth k4 ALONE (card k3) at L12-14, correction .5, 7 held-out nets (k34_spec.py + channel switch); under ~5% closes the question for good. Only if large: a dense MPS replica of a full-depth CP4 bank, bar = beat "exact from L4" (12%).
      - No cheap form carries the unit-specific part (covariance modes 19-27%, scale mixture corr .29-.34, a law on the true k3 17-21%, per-column 4th-order weights 20-24%). The residual (2,2) content needs 256-512 components per layer.
      - Economics: a +20% bill source must cut raw 16.7%; even an exact k3+k4 source nets -2.1%.
      One decisive check remains (analyst): truth k4 ALONE at L12-14, correction .5, 7 held-out nets. Under ~5% closes it for good.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/carry/norm_coupling_mc.py](code/fable_physicist_v1/refit6b/carry/norm_coupling_mc.py)
- [code/fable_physicist_v1/refit6b/carry/col_k4_unit_fit_v2.py](code/fable_physicist_v1/refit6b/carry/col_k4_unit_fit_v2.py)
- [code/fable_physicist_v1/refit6b/carry/k22_rank_census.py](code/fable_physicist_v1/refit6b/carry/k22_rank_census.py)
- [code/fable_analyst_v1/k34_spec.py](code/fable_analyst_v1/k34_spec.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)

<details>
<summary>Additional programs and dependencies (20)</summary>

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
- [code/fable_engineer_v1/mps_lock.py](code/fable_engineer_v1/mps_lock.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L790](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L790)
- [TEAM_LOG_2026-09-22_to_09-26.md#L791](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L791)
- [TEAM_LOG_2026-09-22_to_09-26.md#L792](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L792)
- [TEAM_LOG_2026-09-22_to_09-26.md#L797](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L797)
- [TEAM_LOG_2026-09-22_to_09-26.md#L798](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L798)
- [TEAM_LOG_2026-09-22_to_09-26.md#L799](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L799)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-cheap-fourth-cumulant-sources-miss-per-unit-structure-fail) · [Copy provenance](../../../PROVENANCE.json)

