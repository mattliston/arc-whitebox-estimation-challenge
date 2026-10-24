# Norm-coupling carrier with 16 partner vectors lowers MSE 2.06% on 50 ranks; native cost prevents adoption

2026-10-02T10:43Z · experiment · rejected · measured

The first bill-free carrier of the norm coupling carried the partner vectors of the top-K covariance eigenvectors from layer to layer with matrix-vector products (the ReLU-step law applied to vectors) and read them as an extra (3,1) fourth-cumulant plane (K31) plane. Open loop the carried vectors kept 0.36-0.40 of their energy at depth; closed loop -0.75% / -3.55% on 35 / 651 at K 8, -2.06% (42 of 50) on the 50 held-out ranks at K 16 on top of candidate 3 (K 8: -0.89%), but only -0.8 to -2.3% on six held-out ranks with large per-network scatter, and natively it needs exact eigenvectors (1.4-1.6% of bill without the bank pass, 3.2-3.4% with it). Superseded by the pair-carried law.

Recorded measurement:

    open loop cosine modes 2-8: 0.86-0.91 (layer 2), 0.61-0.79 (5), 0.36-0.76 (9), 0.46-0.74 (13), 0.41-0.71 (14); energy held 0.79 / 0.48 / 0.38 / 0.40 / 0.36; closed loop 35: 0.5 -0.82%, 1.0 -0.75%, 1.5 +0.17%, 2.0 +1.6%, with mode 1 -1.05%; 651: -2.17% / -3.55% / -3.93% / -3.64%, with mode 1 -3.99%; three held-out pooled K 8 -2.1% / -2.9%, K 16 -3.6% / -4.2%; six ranks K 8 modes 2-8 -0.81% (se 0.79), modes 1-8 -1.43% (se 0.86), K 16 -1.9..-2.3% (se 1.3); 50 ranks K 16 -2.057% (se 0.256, 42 of 50), K 8 -0.885% (se 0.202); E38 mode 1 from the bank's trace vector: no gain (35 -0.82% vs -0.75%, 651 -3.15% vs -3.55%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/ncar.py](code/opus_reexam_20261001_v1/ncar.py)
- [code/opus_reexam_20261001_v1/ncar_compare.py](code/opus_reexam_20261001_v1/ncar_compare.py)
- [code/opus_reexam_20261001_v1/ncar_mode1.py](code/opus_reexam_20261001_v1/ncar_mode1.py)
- [code/opus_reexam_20261001_v1/e36_ncar_closed.sh](code/opus_reexam_20261001_v1/e36_ncar_closed.sh)
- [code/opus_reexam_20261001_v1/e37_ncar_fresh.sh](code/opus_reexam_20261001_v1/e37_ncar_fresh.sh)
- [code/opus_reexam_20261001_v1/e38_q1.sh](code/opus_reexam_20261001_v1/e38_q1.sh)
- [code/opus_reexam_20261001_v1/e39_ncar_design.sh](code/opus_reexam_20261001_v1/e39_ncar_design.sh)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (32)</summary>

- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
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
- [code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py](code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py)
- [code/opus_reexam_20261001_v1/k31c.py](code/opus_reexam_20261001_v1/k31c.py)
- [code/opus_reexam_20261001_v1/k31diag.py](code/opus_reexam_20261001_v1/k31diag.py)
- [code/opus_reexam_20261001_v1/k31l.py](code/opus_reexam_20261001_v1/k31l.py)
- [code/opus_reexam_20261001_v1/k31m.py](code/opus_reexam_20261001_v1/k31m.py)
- [code/opus_reexam_20261001_v1/ncar3.py](code/opus_reexam_20261001_v1/ncar3.py)
- [code/opus_reexam_20261001_v1/ncar4.py](code/opus_reexam_20261001_v1/ncar4.py)
- [code/opus_reexam_20261001_v1/ncar5.py](code/opus_reexam_20261001_v1/ncar5.py)
- [code/opus_reexam_20261001_v1/ncar6.py](code/opus_reexam_20261001_v1/ncar6.py)
- [code/opus_reexam_20261001_v1/physicist/f4.py](code/opus_reexam_20261001_v1/physicist/f4.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry.py](code/opus_reexam_20261001_v1/physicist/ncarry.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L241](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L241)
- [TEAM_LOG_2026-10-01_to_10-22.md#L243](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L243)
- [TEAM_LOG_2026-10-01_to_10-22.md#L246](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L246)
- [TEAM_LOG_2026-10-01_to_10-22.md#L251](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L251)
- [TEAM_LOG_2026-10-01_to_10-22.md#L253](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L253)
- [TEAM_LOG_2026-10-01_to_10-22.md#L257](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L257)
- [TEAM_LOG_2026-10-01_to_10-22.md#L268](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L268)
- [TEAM_LOG_2026-10-01_to_10-22.md#L274](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L274)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L511](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L511)

<details>
<summary>All result files and records (3)</summary>

- [NCAR_SPEC.md](NCAR_SPEC.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-norm-coupling-carrier-16-partner-vectors-lowers-mse-2) · [Copy provenance](../../../PROVENANCE.json)

