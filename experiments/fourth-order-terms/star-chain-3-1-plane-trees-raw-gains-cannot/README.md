# Star and chain (3,1)-plane trees: raw gains cannot pay their dense-product cost

2026-09-23T02:36Z · experiment · rejected · measured

Certified star and chain contractions improved raw error but each required three dense matrix products per layer. Their added explanatory power over the isotropic model was small.

Recorded measurement:

    2026-09-22 19:36 analyst: (1) K31 TREES: star+chain certified (matrix = brute force 1e-15); vs 8M MC K31 (from MC-true state) star coef .84-1.08, chain 1.1-1.7, but they add only +2-3% R2 over iso (iso .43-.66). Closed loop (learned ON, pair map only, fable_analyst_v1/k31_tree_closedloop.py via PHYS_SUPPLY fn): tree@1-14 -1.3/-4.1/-0.7/-0.9% (35/100/37/39), star -0.9/-3.2/-1.0/-0.8, chain -0.4/-0.7/-0.2/+0.1, @2-6 +0.5/-1.3/+0.3/+0.1, @11-14 -0.7/-1.0/-1.2/-1.0. Cost star 3n^3 + chain 3n^3 MACs/layer (~1.6% bill/layer each) -> cannot pay. (2) r100 one-step budget done (fable_mc_v1/onestep_budget_r100.log): v_{l+1} card .073/.112/.158% (L7/10/13) -> truth C .042/.057/.074% (-42..-53%), truth K21off -10..-12%, truth K31 -1..-2%; post-mean card 1.14-1.37e-4 -> truth mu 5.3-6.6e-5. BUG in fable_mc_v1/onestep_budget.py row card_state_truth_k3k4: mix() args swapped (it feeds truth K21 off-diag + CARD k3); fixed copy fable_analyst_v1/onestep_budget_fixed.py: truth k3+k4 -> vnext 0%, mean -5..-9%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_analyst_v1/k31_tree_closedloop.py](code/fable_analyst_v1/k31_tree_closedloop.py)
- [code/fable_mc_v1/onestep_budget.py](code/fable_mc_v1/onestep_budget.py)
- [code/fable_analyst_v1/onestep_budget_fixed.py](code/fable_analyst_v1/onestep_budget_fixed.py)
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
<summary>Additional programs and dependencies (15)</summary>

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
- [code/fable_physicist_v1/model_bank_phys.py](code/fable_physicist_v1/model_bank_phys.py)
- [code/fable_physicist_v1/run_supply.py](code/fable_physicist_v1/run_supply.py)
- [code/fable_physicist_v1/supplies.py](code/fable_physicist_v1/supplies.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L21](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L21)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [onestep_budget_r100.log](../../../records/research/2026-09-23-one-step-truth-covariance-supply-removes-4253-next-layer/onestep_budget_r100.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-star-chain-3-1-plane-trees-raw-gains-cannot) · [Copy provenance](../../../PROVENANCE.json)

