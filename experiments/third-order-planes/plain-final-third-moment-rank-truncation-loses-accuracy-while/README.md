# Plain final third-moment rank truncation loses accuracy while pair splitting preserves the read

2026-09-25T19:42Z · experiment · rejected · measured

A direct rank cut damages the final third-moment read because the old tier dominates its contribution. Keeping the pair-difference component full rank while truncating only the pair mean preserves accuracy, but the native cost saving must be measured.

Recorded measurement:

    - 2026-09-25 12:42 lead: L15 OLD-TIER READ (covhunter, torch, 384/320/384, first 10 VALID40 nets): a plain rank cut of the final k3 read FAILS (r = 256 raw +0.87 pt, 192 +3.65, 128 +10.7; dropping the old tier from L15 k3 multiplies MSE by 2.5). The tail matters at L15 because the final k3 feeds the final mean directly. A SPLIT read works: birth pairs p+- = U(f_u +- f_v/2), so k3 is about 3c (U f_u)^2 (U f_v); the common part u is energy-concentrated and truncatable, while the difference part v carries the k3 and needs full rank. u at 256 and v at 384: raw +0.053 pt [-0.007, +0.114], bill about -0.30%, adjusted about -0.25% (L15 only). PARKED as a stand-alone build.
    - 2026-09-25 13:12 covhunter: L15 OLD-TIER k3 READ TRUNCATION (RESULT sec 37; r320_l15split*.py; torch loopR3 params, S28 schedule 384/320/384, VALID40, raw change vs the full read). Plain top-r read: 256 +0.91 pt [+0.65, +1.18]; 192 +3.65; 128 +10.7; old tier dropped from the final k3 +145 pt (MSE x2.4). The plain cut FAILS: the final k3 feeds the final mean directly and the old tier dominates it. SPLIT READ (birth pair p+- = U(f_u +- f_v/2); common part f_u on the first r_u directions, difference part f_v on all 384; tier unchanged): r_u 256 +0.022 pt [-0.009, +0.052]; 192 +0.187 [+0.13, +0.25]. Old-part cost is (r_u + 384)/768 of today's, so on the lead's model (old part 1.8% of bill) r_u 256 gives bill -0.30% and adjusted about -0.28%, borderline vs the -0.3% bar. The native per-direction cost is uncertain (porter map: L15 -0.04% for 32 L13 directions), so a native 2-net bill must decide. The same u truncation applied to the tier at every join costs +1.65 pt on TRAIN24: u-part damage is a propagation effect, v-part damage is direct.
    - 2026-09-25 13:09 lead: L15 READ CLOSED (covhunter RESULT §37, VALID40 40 nets): plain truncation at r = 256 costs +0.91 pt raw (adjusted about +0.29%); the split read at r_u = 256 costs +0.022 pt raw, but its native bill is likely only about -0.07% (porter's map: L15 _l15_rowcut_k3 moves -0.04% per 32 directions), so it's worth well under the bar. KEY ROW for the whole-tier split frames: u truncated to 256 at EVERY join costs +1.65 pt [+1.17, +2.15] on TRAIN24, and VALID40 has run about 1.9x TRAIN24 for these cuts. Against a native bill of -2.0..-2.5%, likely harmful. Covhunter settles it on VALID40 at r_u = 256 and 320; engineer2's prototype is paused until then. The u damage is a propagation effect; the v damage is direct.

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
<summary>Additional programs and dependencies (25)</summary>

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
- [code/fable_covhunter_v1/r320_l15split.py](code/fable_covhunter_v1/r320_l15split.py)
- [code/fable_covhunter_v1/r320_l15split2.py](code/fable_covhunter_v1/r320_l15split2.py)
- [code/fable_covhunter_v1/r320_l15split3.py](code/fable_covhunter_v1/r320_l15split3.py)
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

- [TEAM_LOG_2026-09-22_to_09-26.md#L1225](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1225)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1252](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1252)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1253](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1253)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-plain-final-third-moment-rank-truncation-loses-accuracy-while) · [Copy provenance](../../../PROVENANCE.json)

