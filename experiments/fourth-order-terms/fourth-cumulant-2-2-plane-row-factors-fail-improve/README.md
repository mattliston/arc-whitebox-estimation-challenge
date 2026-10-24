# Fourth-cumulant and (2,2)-plane row factors fail to improve the (3,1)-plane scale

2026-09-23T03:49Z · experiment · rejected · measured

Training-set gains from modifying the (2,2) covariance rows reversed on held-out networks. Better per-unit fourth-cumulant agreement with truth did not yield better final error.

Recorded measurement:

    2026-09-22 20:49 analyst: zero-cost k4/K22 row factors (fable_analyst_v1/rowfactor_closedloop.py via PHYS_SUPPLY; constants chosen on TRAIN 40/41/43/44 only): NONE adds to the physicist's K31 row factor. 35/100/37/39: K31 row alone -1.27/-1.61/-0.19/-3.03%; + K22 x0.9 in the covariance output only (best on TRAIN, -2.5% vs -1.8%) +0.25/-2.75/-0.65/-0.16%; k4 (0.93+0.35 q) with K31 following +0.80/+0.63/+1.03/-0.13%; k4 x0.96 + K31 row -1.01/+0.81/+1.25/-2.63%. K22 changes that reach the radial are a cliff on TRAIN (x0.95 +42%, x0.97 +13%, x1.03 +28%). MC fit: k4 = radial v^2 (0.93 + 0.35 mu k3/v^2) cuts the card's k4 error 15-20% (L14 .27->.22) but the closed loop does not use it. Nothing sent to the porter.
    2026-09-22 21:16 analyst: row package TRAIN sweep (40/41/43/44, torch, learned ON): K31 rows (0.8,0.5) alone -2.03% geo (4/4); + K22 covariance-only rows (0.9,0.5) -2.74% (4/4), (0.9,0.35) -2.72%, (0.95,0.6) -2.53%, (0.85,0.35) -2.46% (3/4); K31 c0 0.85 variants 0.25-0.3 pt weaker. Caveat: on 35/100/37/39 K22-cov x0.9 + K31(.85) was -0.83% vs K31 alone -1.53% (reverses on 35/39). VALID40 of K31(0.8,0.5)+K22c(0.9,0.5) and K31(0.8,0.5) alone running (fable_analyst_v1/rowpackage_valid40.jsonl).
    2026-09-22 21:39 analyst: (1) ROW PACKAGE VALID40 (torch, learned ON): K31(.85,.5) .9856 40/40; K31(.8,.5) .9847 37/40; K31(.8,.5)+K22cov(.9,.5) .9901 31/40 -> K22cov increment 1.0055 (CI 1.0015-1.0097) HARMFUL; porter told not to build it. (2) dC STRUCTURE (fable_analyst_v1/dC_structure.log, dC_propagation.log, dC_swap_v1/v2.log, dC_eigsubspace.log): no cheap structure explains offdiag(P dC P) (<1.2%); 3-layer linear-response inheritance explains 23-47%. Closed-loop swaps at L13 (r35/r100; perp fix .889/.833): top-30 SVD .935/.886, top-100 .913/.843; blocks kink x on .959/.899, on x on .976/.953, kink x kink .980/.991; noise control 1.017/1.029. KEY: the part of the error with at least one side in the top-k eigen-subspace of the CARD's own P C P recovers the lever: k=30 .919/.889 (67-73% of the L13 gain), k=100 .910/.835 (81-99%); at L7+10+13 k=100 .783/.718 vs perp .766/.707 (93-96%). Both-sided k=100 only 36-52%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_analyst_v1/rowfactor_closedloop.py](code/fable_analyst_v1/rowfactor_closedloop.py)
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
<summary>Additional programs and dependencies (12)</summary>

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

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L36](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L36)
- [TEAM_LOG_2026-09-22_to_09-26.md#L45](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L45)
- [TEAM_LOG_2026-09-22_to_09-26.md#L52](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L52)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [rowpackage_valid40.jsonl](rowpackage_valid40.jsonl) · result
- [dC_structure.log](dC_structure.log) · log
- [dC_propagation.log](dC_propagation.log) · log
- [dC_eigsubspace.log](dC_eigsubspace.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-fourth-cumulant-2-2-plane-row-factors-fail-improve) · [Copy provenance](../../../PROVENANCE.json)

