# Half-strength born-unit plane feed plus refit improves 40-network raw error 2.05%

2026-09-25T00:23Z · experiment · adopted · measured

Refitting the unorthogonalized born-unit plane feed recovered its initial learned-on loss and improved held-out error. The gain also held on networks outside the smaller selection set.

Recorded measurement:

    - 2026-09-24 17:23 [physicist] j05 REFIT PASSED ITS BAR. VALID16 1.71227e-8 vs B2 ep1 1.74611e-8 = -1.94% (12/16 better); bar <= 1.7374e-8. From the no-refit start (+4.97%) the refit recovered 6.9%. Extra-epoch control: covhunter's function-preserving wide2x 1-epoch B2 recipe ended at +0.14% and R320 absorbed ~0, so the gain is the j1 physics, not the extra epoch. VALID40 eval launched (s6b_B2ep1_j05_1ep_valid40, 3 threads like the B2 ep1 VALID40 baseline).
    - 2026-09-24 17:40 lead: j05 REFIT PASSES (physicist). VALID16 1.71227e-8 vs B2 ep1 1.74611e-8 = -1.94%, 12/16 (bar 1.7374e-8). It recovered from the +4.97% no-refit start in one epoch (15:50-17:22, no spikes). Extra-epoch controls (wide2x +.14%, R320 ~0) attribute the gain to the j1 PHYSICS. Torch VALID40 is running (~15 min). The porter preps the native port S17-j05 on S16 (the _bd_update j1 leg in physical form at .5 + the params_epoch0 literals); parity, then VALID40 vs S16 and the residual gate. Next physics: second j05 epoch; b x .9 on the j05 card; per-unit k4 on the j05 card. Today's card has NO per-unit k4 channel (k4 = radial*v^2), so nothing competes with the cheap form.
    - 2026-09-24 17:36 [physicist] j05 VALID40 HOLDS: 1.76186e-8 vs B2 ep1 1.79867e-8 = -2.05% (30/40); the 24 nets outside the VALID16 bar -2.11% (18/24); per-net median -2.19%, best -9.02%, worst +4.91% (both eval-only, 3 threads). Porter has the S17-j05 native spec (_bd_update j-loop: j==1 skips the projection, fixed 0.5 in place of c[1]) and params (s6b_B2ep1_j05_1ep/params_epoch0.pt sha256 2000046cdcf2...). Lever 1 one-GEMM per-unit k4, learned-OFF VALID16: 0.9703 (-3.0%, 16/16); learned-ON running; per-band k4 scalar arm (x1.2/1.5/2.0 for L1-7, 8-11, 12-15) queued.
    - 2026-09-24 18:10 lead: j05 VALID40 HOLDS (physicist, eval-only, 3 threads): 1.76186e-8 vs B2 ep1 1.79867e-8 = -2.05%, 30/40; the 24 nets outside VALID16 -2.11% (18/24); per-net median -2.19%. S17 GO for the native build: S16 + view trim + j05 _bd_update change + j05 params + prebind v2b (if gated). Bar: VALID40 vs S16 adjusted <= -1.5%. Expected ~2.80e-9. Next physics: second j05 epoch (bar VALID16 <= 1.7037e-8), b x .9 on the j05 card learned-OFF. The one-GEMM per-unit k4 is -3.0% learned-OFF (16/16) but +9% bill and -0.28% learned-ON (r35): PARKED as a future refit-bundle candidate.
    - 2026-09-24 18:50 lead: S17p VALID40 PASS (porter): adjusted -2.138% [CI -3.09, -1.20] 30/40 vs S16; native j05 matches torch (-2.159% vs -2.046%, per-net corr .996); expected ~2.79e-9; staging after the residual gate. Star torch VALID40 at 114/122: B2 r32 +.35%, r64 +.27%, j05 r32 +.20%, so S18 (S17p + low-rank star, -5.6% bill) is on track. COVHUNTER: the j03 refit passes (-1.65% vs B2 ep1) but is +.30% vs j05, so j05 is the better strength. The L1 dials' shadow is REDUNDANT learned ON (no-refit +7.87%, full shadow +5.70%, beta* .30 -> -1.30%). The joint refit (launched 18:28) must recover ~8-9 pts, a stretch. K31<->j1 are SUBSTITUTES (the fit moves j1 up and c0 down together; the K31 row factor was partly j1's stand-in). GO: LEARNED-ON joint GN fit of the 12 dials on j05 ep0 (3 threads). If <= -1% no refit, it is a literal-only ship.

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
<summary>Additional programs and dependencies (18)</summary>

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
- [code/fable_physicist_v1/refit6b/refit_s6b.py](code/fable_physicist_v1/refit6b/refit_s6b.py)
- [code/fable_physicist_v1/refit6b/s6bcard_j05.py](code/fable_physicist_v1/refit6b/s6bcard_j05.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L508](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L508)
- [TEAM_LOG_2026-09-22_to_09-26.md#L509](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L509)
- [TEAM_LOG_2026-09-22_to_09-26.md#L519](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L519)
- [TEAM_LOG_2026-09-22_to_09-26.md#L520](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L520)
- [TEAM_LOG_2026-09-22_to_09-26.md#L551](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L551)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332224: half-strength unit birth feed, refit, lean views, and setup prebinding](../../../submissions/332224/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-half-strength-born-unit-plane-feed-plus-refit-improves) · [Copy provenance](../../../PROVENANCE.json)

