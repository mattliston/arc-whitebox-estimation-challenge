# Joint norm-coupling dial solve lowers MSE 1.48% on 25 ranks and enters submission #333587

2026-10-02T15:32Z · experiment · adopted · measured

The pair law's own dials were solved jointly from single-change forward runs on 16 line-search ranks (candidate generation 39 (S39) card, table, three forms): fold strength 1.829 at layers 1-4, 1.395 at 5-9, 1.181 at 10-14, factor 1.272 on the removed multiple of C, source strengths births 1.142, rows 0.697, diag 0.791, plus the card's (3,1) fourth-cumulant plane (K31) row factor c0 0.6222 and k31 knob -0.1659. Judged on the first 25 ranks against default dials: -1.48%. Natively the values gain with the table (-2.51% / +0.20% on 651 / 378) but not with the constant regressor, because the factor was solved on the table's falling multiples; candidate generation 41 (S41) kept the table.

Recorded measurement:

    model -1.80% (leave-one-out -1.30%), real run -1.43% (se 0.31, 14 of 16); 25 ranks: 1.4464e-8 vs 1.4681e-8 = -1.479% (se 0.283, 20 of 25), all layers -0.765%; shifts: c0 +0.088, k31 +0.046, n6b +0.142, n6d -0.209, n6r -0.303, nfd +0.272, nfs 1-4 +0.829, 5-9 +0.395, 10-14 +0.181; native 651 / 378 with table -2.51% / +0.20% (pooled -1.16%), with constant regressor -0.06% / +0.42%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/engineer/a38_pairlaw/dials_from_trainer.py](code/opus_reexam_20261001_v1/engineer/a38_pairlaw/dials_from_trainer.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)

<details>
<summary>Additional programs and dependencies (34)</summary>

- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/ncar.py](code/opus_reexam_20261001_v1/ncar.py)
- [code/opus_reexam_20261001_v1/ncar4.py](code/opus_reexam_20261001_v1/ncar4.py)
- [code/opus_reexam_20261001_v1/ncar5.py](code/opus_reexam_20261001_v1/ncar5.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar6_audit_frozen_s3d.py](code/opus_reexam_20261001_v1/physicist/ncar6_audit_frozen_s3d.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_fold_frozen_s2w.py](code/opus_reexam_20261001_v1/physicist/ncar_fold_frozen_s2w.py)
- [code/opus_reexam_20261001_v1/physicist/ncarry.py](code/opus_reexam_20261001_v1/physicist/ncarry.py)
- [code/opus_reexam_20261001_v1/physicist/pbase_frozen_s4a.py](code/opus_reexam_20261001_v1/physicist/pbase_frozen_s4a.py)
- [code/opus_reexam_20261001_v1/trainer/cert/fold/dial_verdict.py](code/opus_reexam_20261001_v1/trainer/cert/fold/dial_verdict.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508.py](code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508y.py](code/opus_reexam_20261001_v1/trainer/frozen/k31l_t1508y.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1339.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1339.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402x.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402x.py)
- [code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402y.py](code/opus_reexam_20261001_v1/trainer/frozen/ncar6_t1402y.py)
- [code/opus_reexam_20261001_v1/trainer/m_mix.py](code/opus_reexam_20261001_v1/trainer/m_mix.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/opus_reexam_20261001_v1/trainer/worker.py](code/opus_reexam_20261001_v1/trainer/worker.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L354](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L354)
- [TEAM_LOG_2026-10-01_to_10-22.md#L361](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L361)
- [TEAM_LOG_2026-10-01_to_10-22.md#L366](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L366)
- [TEAM_LOG_2026-10-01_to_10-22.md#L369](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L369)
- [RESULT.md#L877](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md#L877)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333587: young-pair fourth-chaos plane, norm-coupling dial solve and fewer kernel requests](../../../submissions/333587/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-joint-norm-coupling-dial-solve-lowers-mse-1-48) · [Copy provenance](../../../PROVENANCE.json)

