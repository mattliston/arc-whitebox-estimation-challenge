# True norm-coupling matrix as a (3,1) plane lowers MSE 7.99% / 15.60% on networks 35 / 651

2026-10-02T09:28Z · experiment · informational · measured

The true network-specific part of the norm-coupling matrix at layers 1-15 (2 x 4M Monte Carlo samples per network) was injected into the candidate generation 38 (S38) physics on the research card through its three possible uses. As an extra (3,1) fourth-cumulant plane (K31) plane (3/n) R through the pair map's remainder input it gives -7.99% / -15.60% on networks 35 / 651 (rank 64: -7.27% / -14.72%, rank 32: -6.07% / -15.03%); as a feed plane -1.09% / +2.15%; as per-unit kurtosis on top of the kurtosis term +3.23% / -0.95%, and all three together are worse than the plane alone. By the lead's pre-set rule (all three at rank 64 >= 4%) the answer was to build a carrier, the whole value being the K31 plane; the oracle ceiling on six held-out ranks is -6.46%.

Recorded measurement:

    base 13.0215e-9 (35), 15.6222e-9 (651); K31 plane full / rank 64 / rank 32: -7.99% / -7.27% / -6.07% (35), -15.60% / -14.72% / -15.03% (651); feed plane -1.09% / -1.52% / -1.77% (35), +2.15% / +1.41% / +1.99% (651); per-unit k4 +3.23% / +2.50% / +1.36% (35), -0.95% / +1.88% / -1.40% (651); all three -2.26% / -3.97% / -5.15% (35), -14.80% / -12.00% / -14.46% (651); oracle on six held-out ranks -6.46%; block agreement of the network-specific part 0.93-0.98 at layers 5-15

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/physicist/e16b_f4b_frozen_s2e.py](code/opus_reexam_20261001_v1/physicist/e16b_f4b_frozen_s2e.py)
- [code/opus_reexam_20261001_v1/physicist/f4.py](code/opus_reexam_20261001_v1/physicist/f4.py)
- [code/opus_reexam_20261001_v1/physicist/f4b_frozen_fr24k.py](code/opus_reexam_20261001_v1/physicist/f4b_frozen_fr24k.py)
- [code/opus_reexam_20261001_v1/physicist/ntor_frozen_s2e.py](code/opus_reexam_20261001_v1/physicist/ntor_frozen_s2e.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)
- [code/opus_reexam_20261001_v1/physicist/so2_frozen_s2b.py](code/opus_reexam_20261001_v1/physicist/so2_frozen_s2b.py)
- [code/opus_reexam_20261001_v1/physicist/t19g_mc_all.py](code/opus_reexam_20261001_v1/physicist/t19g_mc_all.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L221](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L221)
- [TEAM_LOG_2026-10-01_to_10-22.md#L222](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L222)
- [TEAM_LOG_2026-10-01_to_10-22.md#L229](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L229)
- [TEAM_LOG_2026-10-01_to_10-22.md#L232](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L232)
- [TEAM_LOG_2026-10-01_to_10-22.md#L250](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L250)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L500](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L500)
- [RESULT.md#L1202](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md#L1202)

<details>
<summary>All result files and records (4)</summary>

- [t4_ora.log](t4_ora.log) · log
- [RESULT.md](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-true-norm-coupling-matrix-3-1-plane-lowers-mse) · [Copy provenance](../../../PROVENANCE.json)

