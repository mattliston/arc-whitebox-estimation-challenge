# Joint physics fit with learned corrections off gains 10.06% but is mostly redundant when restored

2026-09-25T01:15Z · experiment · inconclusive · measured

A twelve-dial Gauss-Newton fit strongly improved the learned-off estimator. Restoring learned corrections made those values harmful, and a shadow test showed substantial redundancy.

Recorded measurement:

    2026-09-24 18:15 covhunter: L1 JOINT LEARNED-OFF GN FIT PASSES BIG (RESULT sec 25): j05 card + j05 ep0 params, learned OFF, 12 physics dials (j1/j0/j2/j3 physical, b scale, carried/newborn lam scales, K31 c0/c1, k4 x3 bands), 1 FD run/dial on 8 TRAIN nets. Fit: in-sample -11.86%, LOO -10.83% (8/8), REAL run -11.89% (8/8). VALID16 learned-OFF vs j05 off-control 2.02995e-8 (= physicist's to -1e-5): 1.82580e-8 = -10.06% pooled, 16/16 (se 1.1%). Dials: X1 .683, X2 +.295, X3 -.415, c0 .85->.458, c1 .40, k4 x.954/.953/.915, s_carry 1.0076/layer, s_new .987, b 1.026. Shadow law says ~0.3-0.5x survives learned-ON -> next: shadow test (12 min) + j05+joint refit.
    - 2026-09-24 18:40 lead: *** COVHUNTER L1 JOINT LEARNED-OFF GAUSS-NEWTON FIT ON THE j05 CARD: VALID16 -10.06% vs the j05 learned-off control, 16/16 (per-net se 1.07%). Fit on 8 TRAIN nets: GN model -11.86%, LOO -10.83% (8/8), real run -11.89%. Fitted dials (physical units):
      - K31 row factor c0 .85->.458, c1 .5->.396;
      - k4 x.954/.953/.915 (L1-7/8-11/12-15);
      - carried content x1.0076/layer, newborn x.987; b x1.026;
      - j1 .683, j2 +.295, j3 -.415, j0 -.061.
      Shadow law says ~.3-.5x survives learned ON, i.e. -3..-5% after a refit. GO: (1) shadow test + learned-ON no-refit value now; (2) j05+joint refit (1 epoch, from j05 ep0, dials frozen) when j03 exits (~18:35). Bar VALID16 <= 1.6951e-8 (-1% vs j05 ep0).
    2026-09-24 18:27 covhunter: j05+JOINT REFIT LAUNCHED (lead GO): fable_covhunter_v1/jfit1_j05ep0_1ep (log jfit1_j05ep0_1ep.log), card s6bcard_jfit1.py = j05 card + L1 dials frozen as literals (physical units: K31 row c0 .85->.458, c1 .5->.396; k4 x .954/.953/.915 on L1-7/8-11/12-15; carried lam x1.0076 per layer, newborn x.987; b x1.026; j1 .683, j2 +.295, j3 -.415, j0 -.061 physical). Certified: m(OFF, fit) == the fit's VALID16 learned-OFF per net (bit-exact, 16/16). B2 recipe (full+P, lr 1e-4/5e-5, warmup 3/5, clip 1), 1 epoch TRAIN160, warm j05 ep0, VALID16, j05 env (BLAS 2, torch 5, nice 10). PRE-REGISTERED BAR: VALID16 <= 1.6951e-8 (-1% vs j05 ep0 1.71227e-8); VALID40 if it passes.
    2026-09-24 18:28 covhunter: (1) j03 REFIT ep0: VALID16 1.71734e-8 = -1.65% vs B2 ep1 (11/16), PASS its bar 1.7374e-8; +0.30% vs j05 ep0 (6/16) -> j05 is the better strength. (2) SHADOW OF THE L1 DIALS (j05 ep0 ON base 1.71226e-8): module cert 16/16 bit-exact; learned-ON no-refit with the fitted dials +7.87% (0/16); shadow beta=1 +5.70% (1/16), beta* .30 -> -1.30%: the learned-OFF -10% is mostly REDUNDANT with the learned parts; the refit must recover ~8 pt to break even, ~9 to hit -1% (j05's refit recovered 6.9 pt/epoch). (3) j05+JOINT REFIT jfit1_j05ep0_1ep LAUNCHED 18:28 (bar VALID16 <= 1.6951e-8).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_covhunter_v1/s6bcard_jfit1.py](code/fable_covhunter_v1/s6bcard_jfit1.py)
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
<summary>Additional programs and dependencies (17)</summary>

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
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L540](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L540)
- [TEAM_LOG_2026-09-22_to_09-26.md#L541](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L541)
- [TEAM_LOG_2026-09-22_to_09-26.md#L542](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L542)
- [TEAM_LOG_2026-09-22_to_09-26.md#L543](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L543)
- [TEAM_LOG_2026-09-22_to_09-26.md#L544](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L544)
- [TEAM_LOG_2026-09-22_to_09-26.md#L545](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L545)
- [TEAM_LOG_2026-09-22_to_09-26.md#L546](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L546)
- [TEAM_LOG_2026-09-22_to_09-26.md#L549](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L549)
- [TEAM_LOG_2026-09-22_to_09-26.md#L550](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L550)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [jfit1_j05ep0_1ep.log](jfit1_j05ep0_1ep.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-joint-physics-fit-learned-corrections-off-gains-10-06) · [Copy provenance](../../../PROVENANCE.json)

