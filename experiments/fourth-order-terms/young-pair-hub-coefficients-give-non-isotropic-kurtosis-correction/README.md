# Young-pair hub coefficients give a non-isotropic kurtosis correction and 1.41% further error reduction

2026-10-02T01:35Z · experiment · adopted · measured

The card's per-unit fourth cumulant is radial v^2, 14-26% rms off per unit against 32M truth. K4L (the lead's per-unit kurtosis term) builds a non-isotropic correction elementwise from the exact hub coefficients S1..S4 of each young column pair (the transported hub vector and covariance column), keeps the card's isotropic level and adds c times the deviation. The screen four read -2.83% alone and -5.50% with F4, but on 24 held-out ranks K4L on top of F4 is -1.41% (18 of 24), half the screen value, with age <= 4 worse than age <= 2; the native port reproduces it network by network at +0.115% bill. Shipped in S37 at age <= 2, c = 1.

Recorded measurement:

    card k4 relative rms error per unit 0.138 / 0.208 / 0.221 / 0.261 at layers 5 / 9 / 11 / 13; open loop residual explained 0.28 / 0.26 / 0.25 / 0.34 (coefficient 0.88 / 0.89 / 0.88 / 1.02); screen four K4L alone -2.83%, F4+K4L -5.50% (-7.43 / -3.64 / -8.24 / -3.16); validation 42 / 48 / 59 / 63: +1.16 / -2.32 / +2.54 / -4.79%; 24 held-out: F4+K4L -4.90% [-5.66,-4.13] 24/24, K4L increment -1.41% [-2.22,-0.59] (mean -1.34%, se 0.41, 18 of 24, worst +2.9%), age <= 4 -0.87% [-1.90,+0.25]; native parity pooled -2.61% vs torch -2.83% (K4L alone), -5.43% vs -5.50% (F4+K4L); native 10 fresh K4L increment -0.86% (7 of 10), torch -0.69%, correlation 0.990; bill +0.115% (K4L), +0.340% (F4+K4L), +1,775 charged ops

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/e16_local_k4.py](code/opus_reexam_20261001_v1/e16_local_k4.py)
- [code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py](code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py)
- [code/opus_reexam_20261001_v1/e16c_fresh24.sh](code/opus_reexam_20261001_v1/e16c_fresh24.sh)
- [code/opus_reexam_20261001_v1/e16d_k4_by_age.py](code/opus_reexam_20261001_v1/e16d_k4_by_age.py)
- [code/opus_reexam_20261001_v1/engineer/a15_k4l/fragment_k4l_v2.py](code/opus_reexam_20261001_v1/engineer/a15_k4l/fragment_k4l_v2.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)

<details>
<summary>Additional programs and dependencies (30)</summary>

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
- [code/opus_reexam_20261001_v1/k31c.py](code/opus_reexam_20261001_v1/k31c.py)
- [code/opus_reexam_20261001_v1/k31diag.py](code/opus_reexam_20261001_v1/k31diag.py)
- [code/opus_reexam_20261001_v1/k31l.py](code/opus_reexam_20261001_v1/k31l.py)
- [code/opus_reexam_20261001_v1/k31m.py](code/opus_reexam_20261001_v1/k31m.py)
- [code/opus_reexam_20261001_v1/ncar.py](code/opus_reexam_20261001_v1/ncar.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L67](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L67)
- [TEAM_LOG_2026-10-01_to_10-22.md#L69](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L69)
- [TEAM_LOG_2026-10-01_to_10-22.md#L72](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L72)
- [TEAM_LOG_2026-10-01_to_10-22.md#L76](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L76)
- [TEAM_LOG_2026-10-01_to_10-22.md#L84](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L84)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L325](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L325)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L341](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L341)

<details>
<summary>All result files and records (5)</summary>

- [K4L_SPEC.md](K4L_SPEC.md) · record
- [e16_rank0035.log](e16_rank0035.log) · log
- [e16d_rank0035.log](e16d_rank0035.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- shipped in: [Submission #333435: fourth-cumulant column shifts and per-unit kurtosis for pairs up to age two](../../../submissions/333435/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-young-pair-hub-coefficients-give-non-isotropic-kurtosis-correction) · [Copy provenance](../../../PROVENANCE.json)

