# Young-column fourth-cumulant shifts reduce held-out error by 3.54% on all 24 networks

2026-10-01T23:45Z · experiment · adopted · measured

Every bank column stores one scalar from its birth, A = -alpha / (sigma (1 + radial/3)) of its hub, so the columns carry the Gaussian-born fourth-order star at no cost and its feed into the next births is an in-place shift of the young carried columns after the gate, g o pp += theta A (d/2) o pp o pp (columns at most two layers old), with a cheap anchor that removes the added tensor's plane from the star (F4). Torch without refit: -2.33% on the four screen networks, -2.46% on VALID40, -3.54% on 24 held-out newer ranks (24 of 24); the native port reproduces torch network by network (-3.83% on 20 fresh ranks, -1.695% bill with the strip stack); strength 0.6 beat 0.4 by 0.35-0.52% on fresh ranks and was shipped in S37.

Recorded measurement:

    torch no refit: screen four -2.33% (theta 0.4), learned off -4.07%, shadow -2.85%; VALID40 theta 0.2 -1.67% [-1.99,-1.34] 39/40, theta 0.4 -2.46% [-3.03,-1.85] 35/40, all-layer -2.22%; 24 held-out -3.54% [-4.10,-2.95] 24/24, all-layer -3.00%; theta 0.6 vs 0.4: -0.52% (se 0.22, 11 of 17), -0.35% [-0.70,-0.00] on 24; theta 0.8 vs 0.4 -0.06%/+0.16%; native parity 35 / 100: -4.48% / +0.73% vs torch -4.25% / +0.95%, cosine 0.955-0.968; native 20 fresh: MSE -3.83% [-4.46,-3.15] 19 of 20, bill -1.695%, adjusted -5.47%; bill +0.225% (16 flops per element of live rows x shifted columns, +690 charged ops); generated seeds: F4 0.4 -3.22% (8 of 8)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/f4.py](code/opus_reexam_20261001_v1/physicist/f4.py)
- [code/opus_reexam_20261001_v1/physicist/t3_screen.py](code/opus_reexam_20261001_v1/physicist/t3_screen.py)
- [code/opus_reexam_20261001_v1/physicist/f4_cert.py](code/opus_reexam_20261001_v1/physicist/f4_cert.py)
- [code/opus_reexam_20261001_v1/engineer/a14_f4/fragment_f4_v2.py](code/opus_reexam_20261001_v1/engineer/a14_f4/fragment_f4_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a14_f4/parity_torch.py](code/opus_reexam_20261001_v1/engineer/a14_f4/parity_torch.py)
- [code/opus_reexam_20261001_v1/engineer/a14_f4/fresh_compare.py](code/opus_reexam_20261001_v1/engineer/a14_f4/fresh_compare.py)
- [code/opus_reexam_20261001_v1/e16e_fresh24_theta.sh](code/opus_reexam_20261001_v1/e16e_fresh24_theta.sh)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (34)</summary>

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
- [code/opus_reexam_20261001_v1/ncar.py](code/opus_reexam_20261001_v1/ncar.py)
- [code/opus_reexam_20261001_v1/ncar3.py](code/opus_reexam_20261001_v1/ncar3.py)
- [code/opus_reexam_20261001_v1/ncar4.py](code/opus_reexam_20261001_v1/ncar4.py)
- [code/opus_reexam_20261001_v1/ncar5.py](code/opus_reexam_20261001_v1/ncar5.py)
- [code/opus_reexam_20261001_v1/ncar6.py](code/opus_reexam_20261001_v1/ncar6.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry.py](code/opus_reexam_20261001_v1/physicist/ncarry.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)
- [code/opus_reexam_20261001_v1/physicist/t2_lib.py](code/opus_reexam_20261001_v1/physicist/t2_lib.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L46](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L46)
- [TEAM_LOG_2026-10-01_to_10-22.md#L59](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L59)
- [TEAM_LOG_2026-10-01_to_10-22.md#L62](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L62)
- [TEAM_LOG_2026-10-01_to_10-22.md#L68](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L68)
- [TEAM_LOG_2026-10-01_to_10-22.md#L71](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L71)
- [TEAM_LOG_2026-10-01_to_10-22.md#L77](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L77)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L265](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L265)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L349](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L349)

<details>
<summary>All result files and records (5)</summary>

- [t4_valid40_v40a.log](t4_valid40_v40a.log) · log
- [t4_fr24a.log](t4_fr24a.log) · log
- [RESULT.md](../../../records/research/2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- shipped in: [Submission #333435: fourth-cumulant column shifts and per-unit kurtosis for pairs up to age two](../../../submissions/333435/README.md)

[Research graph](../../../graph/index.html#node=2026-10-01-young-column-fourth-cumulant-shifts-reduce-held-out-error) · [Copy provenance](../../../PROVENANCE.json)

