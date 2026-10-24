# Young- and old-plane read scales jointly fitted: final MSE falls 0.226% on 25 ranks

2026-10-02T17:25Z · experiment · inconclusive · measured

Exact gradients and finite differences support a small reduction in young-column read weights, opposite to the true plane-size deficit. A joint solve of young scales by layer band and the old-column scale lowers final-layer MSE 0.226% on 25 ranks, with all-layer error 0.457% higher.

Recorded measurement:

    young +0.01 scale: finite difference +2.68% versus first-order +1.38%; derivative correlation 0.99; old control correlation 1.00; 25 ranks 1.4237e-08 vs 1.4269e-08, -0.226% (se 0.143), 13/25; all layers +0.457%; old shift +0.006, young layers 1-4 / 5-9 -0.006, layers 10-14 +0.003

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/trainer/cert/fold/tier_grad_table.py](code/opus_reexam_20261001_v1/trainer/cert/fold/tier_grad_table.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L393](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L393)
- [TEAM_LOG_2026-10-01_to_10-22.md#L405](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L405)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [spec_tier.json](spec_tier.json) · config

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-young-old-plane-read-scales-jointly-fitted-final-mse) · [Copy provenance](../../../PROVENANCE.json)

