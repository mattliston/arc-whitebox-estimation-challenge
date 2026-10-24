# Norm-coupling law on column pairs lowers MSE 5.80% on 50 ranks; native implementation enters submission #333559

2026-10-02T13:40Z · experiment · adopted · derived

The card's newborn bank columns of a hub are p+- = s_h + t+- e_h with s_h = g o Cpre[:, h], and the law's fresh sources (births S2 c c', row forms rv (e c' + c e'), diagonal D e e') use the same c_h, so every hub's contribution is a symmetric 2 x 2 form on its own pair and the bank's transport carries it: one second weight per column plus one cross weight per pair, no eigenvectors, probes or rank cut. Read natively as a richer left factor of the (2,1) third-cumulant plane (K21) bank read with the isotropic multiple of C removed (by a per-layer table times a diagonal regression, later a constant regressor) it gives -14.43% on held-out 651 (full-matrix law -15.1%), -5.60% on seven networks with the table, -5.80% (one form) and -6.24% (three forms) on the 50 judging ranks, and natively -7.50% (parity version) / -7.93% (fired file) on ten fresh networks for +0.88% bill. Shipped in candidate generation 40 (S40) with three forms, the card's beta and all sources at strength 1.

Recorded measurement:

    651 (base 16.008e-9): bank channel -4.64%, + row forms -12.44%, + diagonal -12.36%, + births -14.43%, without isotropic-rest sandwich -14.90%; fold four exact forms -13.84%, multiple of C only -12.63%, strength 1.25 -14.65%; seven networks: reference plane -7.49%, fold four forms -7.46%, multiple of C exact -5.78%, table -5.60%; constant regressor one form -5.81%, three forms -6.13%; 50 ranks on candidate 3: one form -5.804% (se 0.499, 47 of 50), three forms -6.237% (se 0.508, 50 of 50), four exact forms on 25 -7.627%; native parity version ten fresh vs S39: -7.50% [-9.45,-5.49], 10 of 10, bill +1.168%, MSE x bill -6.46%; fired spec file: -7.93% [-9.97,-5.95], bill +0.876%, MSE x bill -7.16%; native = torch network by network (287 -3.23 / -3.2, 328 -8.18 / -7.6, 378 -6.51 / -6.3, 424 -10.63 / -10.4, 641 -1.85 / -1.5, 651 -13.04 / -13.8); lean port +871 / +885 logged calls (parity +3,568); unit check plane 3-9e-5 of the fold term

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/ncar6.py](code/opus_reexam_20261001_v1/ncar6.py)
- [code/opus_reexam_20261001_v1/e51_pairlaw.sh](code/opus_reexam_20261001_v1/e51_pairlaw.sh)
- [code/opus_reexam_20261001_v1/e51_summary.py](code/opus_reexam_20261001_v1/e51_summary.py)
- [code/opus_reexam_20261001_v1/e52_pairlaw_ablate.sh](code/opus_reexam_20261001_v1/e52_pairlaw_ablate.sh)
- [code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v1.py](code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v4.py](code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v4.py)
- [code/opus_reexam_20261001_v1/engineer/a38_pairlaw/unit_np.py](code/opus_reexam_20261001_v1/engineer/a38_pairlaw/unit_np.py)
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
- [code/opus_reexam_20261001_v1/ncar.py](code/opus_reexam_20261001_v1/ncar.py)
- [code/opus_reexam_20261001_v1/ncar3.py](code/opus_reexam_20261001_v1/ncar3.py)
- [code/opus_reexam_20261001_v1/ncar4.py](code/opus_reexam_20261001_v1/ncar4.py)
- [code/opus_reexam_20261001_v1/ncar5.py](code/opus_reexam_20261001_v1/ncar5.py)
- [code/opus_reexam_20261001_v1/physicist/f4.py](code/opus_reexam_20261001_v1/physicist/f4.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry.py](code/opus_reexam_20261001_v1/physicist/ncarry.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L312](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L312)
- [TEAM_LOG_2026-10-01_to_10-22.md#L320](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L320)
- [TEAM_LOG_2026-10-01_to_10-22.md#L325](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L325)
- [TEAM_LOG_2026-10-01_to_10-22.md#L327](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L327)
- [TEAM_LOG_2026-10-01_to_10-22.md#L329](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L329)
- [TEAM_LOG_2026-10-01_to_10-22.md#L332](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L332)
- [TEAM_LOG_2026-10-01_to_10-22.md#L335](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L335)
- [TEAM_LOG_2026-10-01_to_10-22.md#L342](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L342)
- [TEAM_LOG_2026-10-01_to_10-22.md#L344](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L344)
- [TEAM_LOG_2026-10-01_to_10-22.md#L350](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L350)
- [TEAM_LOG_2026-10-01_to_10-22.md#L356](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L356)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L546](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L546)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L553](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L553)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L556](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L556)
- [RESULT.md#L1367](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L1367)

<details>
<summary>All result files and records (6)</summary>

- [NCAR6_SPEC.md](NCAR6_SPEC.md) · record
- [rank0651_e48.log](rank0651_e48.log) · log
- [rank0651_e49.log](rank0651_e49.log) · log
- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- shipped in: [Submission #333559: norm-coupling law on column pairs folded into the third-order plane read](../../../submissions/333559/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-norm-coupling-law-column-pairs-lowers-mse-5-80) · [Copy provenance](../../../PROVENANCE.json)

