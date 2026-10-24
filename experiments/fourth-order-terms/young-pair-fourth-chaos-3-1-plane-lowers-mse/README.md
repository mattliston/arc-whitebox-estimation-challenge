# Young-pair fourth-chaos (3,1) plane lowers MSE 1.87% on 25 ranks and enters submission #333587

2026-10-02T15:17Z · experiment · adopted · derived

The per-unit kurtosis term uses the young pairs' local quartic form on the diagonal only and the pair law carries its second-chaos projection; the fourth-chaos rest of Q(i,i,i,j) was in no plane. Folded into the same read for cohorts of age 1-2 at strength 1 with nothing removed it adds 1.5 points on seven networks (six of seven better) and -1.87% on the first 25 judging ranks on top of the law with the constant regressor; the truth planes confirm strength 1 with no double count (least-squares coefficients 0.86-1.03). The native port is a cubic-plus-linear polynomial per column with six coefficients from one table (parity -2.15% / -2.58% vs torch -2.09% / -2.65%), lean form +0.29% bill; the age-3 path is worth about 0.2%.

Recorded measurement:

    seven networks: law -6.13% -> law + K31L4 -7.60% (per network -6.2 -3.4 -7.7 -8.3 -12.3 -1.5 -14.7 vs -6.2 -1.8 -7.1 -5.8 -9.0 -0.9 -13.0), all layers -7.99% vs -6.29%; 651: remainder31 -1.96%, folded -2.09%, age 1 only +0.96%, ages <= 3 / 4 -1.23 / -1.51%, strength 0.5 / 1.5 -1.38 / -1.52%; 25 ranks: -1.867% (se 0.388, 21 of 25) vs law, -8.180% vs candidate 3; column means over all units -2.09 / -3.04% (651 / 378), response rows only -1.97 / -2.88%, without cross term -2.11 / -3.07%; truth coefficients (law, age 1, age 2) 1.01 / 1.00 / 1.02 at layer 2, 0.85 / 0.86 / 0.86 at layer 14; native parity 651 / 378: -2.15% / -2.58% vs torch -2.09% / -2.65%, bill +0.388% (v6), lean v7 +0.292%, +479 / +475 calls; age-3 path at 0.72: -0.267% (se 0.121)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/k31l.py](code/opus_reexam_20261001_v1/k31l.py)
- [code/opus_reexam_20261001_v1/e62_k31l4.sh](code/opus_reexam_20261001_v1/e62_k31l4.sh)
- [code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v6.py](code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v6.py)
- [code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v7.py](code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v7.py)
- [code/opus_reexam_20261001_v1/engineer/a39_k31l/identity_check.py](code/opus_reexam_20261001_v1/engineer/a39_k31l/identity_check.py)
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
- [code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py](code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py)
- [code/opus_reexam_20261001_v1/k31c.py](code/opus_reexam_20261001_v1/k31c.py)
- [code/opus_reexam_20261001_v1/k31diag.py](code/opus_reexam_20261001_v1/k31diag.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L352](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L352)
- [TEAM_LOG_2026-10-01_to_10-22.md#L355](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L355)
- [TEAM_LOG_2026-10-01_to_10-22.md#L358](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L358)
- [TEAM_LOG_2026-10-01_to_10-22.md#L362](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L362)
- [TEAM_LOG_2026-10-01_to_10-22.md#L363](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L363)
- [TEAM_LOG_2026-10-01_to_10-22.md#L364](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L364)
- [TEAM_LOG_2026-10-01_to_10-22.md#L388](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L388)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L588](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L588)
- [RESULT.md#L1512](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L1512)

<details>
<summary>All result files and records (4)</summary>

- [NCAR6_SPEC.md](../norm-coupling-law-column-pairs-lowers-mse-5-80/NCAR6_SPEC.md) · record
- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- shipped in: [Submission #333587: young-pair fourth-chaos plane, norm-coupling dial solve and fewer kernel requests](../../../submissions/333587/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-young-pair-fourth-chaos-3-1-plane-lowers-mse) · [Copy provenance](../../../PROVENANCE.json)

