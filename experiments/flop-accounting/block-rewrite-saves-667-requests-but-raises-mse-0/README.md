# Block rewrite saves 667 requests but raises MSE 0.26%; byte-identical replacement saves 323 requests

2026-10-02T16:57Z · experiment · rejected · measured

Rewriting the exact-term module's step and the kurtosis term's hub terms as single block operations saves 667-670 requests per network but costs +0.073% of bill (block copies replace per-cohort views) and moves the prediction at the rounding floor (4-5e-6 rms); on ten fresh networks it read MSE +0.264% (8 of 10 higher) against the first two networks' -0.61% / -1.26%, so it was withdrawn as a default. A byte-identical diet took its place: column blocks by splits, two-operand einsums, the edge cut's kept blocks gathered by one index each: -323 requests on every network, bill -0.023%, predictions equal bit for bit. Rule: a rounding-level rewrite needs ten networks before it is offered.

Recorded measurement:

    deep diet 651 / 378: -667 / -670 requests, bill +0.0775% / +0.0790%, prediction 5.6 / 4.6e-6 rms, MSE -0.61% / -1.26%; ten fresh: MSE +0.264% [-0.157,+0.599] (8 of 10 higher), all layers +0.17%, bill +0.073%, MSE x bill +0.33%; pieces: F4c v2 -416 (+0.039%), K4L v6 -211 (+0.036%), pair law v9 -45 to -56 (+0.001%); byte-identical diet: -323 / -323 requests (logged +22, index calls -345), bill -0.0233% / -0.0235%; edge cut v2 41 -> 34 requests a layer, -9 Mflop a layer

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
<summary>Additional programs and dependencies (23)</summary>

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
- [code/opus_reexam_20261001_v1/engineer/a15_k4l/fragment_k4l_v6.py](code/opus_reexam_20261001_v1/engineer/a15_k4l/fragment_k4l_v6.py)
- [code/opus_reexam_20261001_v1/engineer/a15_k4l/fragment_k4l_v7.py](code/opus_reexam_20261001_v1/engineer/a15_k4l/fragment_k4l_v7.py)
- [code/opus_reexam_20261001_v1/engineer/a15_k4l/unit_k4l_v6.py](code/opus_reexam_20261001_v1/engineer/a15_k4l/unit_k4l_v6.py)
- [code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut_v2.py](code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a19_edge_cut/unit_ecut_v2.py](code/opus_reexam_20261001_v1/engineer/a19_edge_cut/unit_ecut_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a27_f4c/fragment_f4c_v2.py](code/opus_reexam_20261001_v1/engineer/a27_f4c/fragment_f4c_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a27_f4c/fragment_f4c_v3.py](code/opus_reexam_20261001_v1/engineer/a27_f4c/fragment_f4c_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a27_f4c/unit_f4c_v2.py](code/opus_reexam_20261001_v1/engineer/a27_f4c/unit_f4c_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v9.py](code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v9.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/f4c_native_ref.py](code/opus_reexam_20261001_v1/physicist/f4c_native_ref.py)
- [code/opus_reexam_20261001_v1/physicist/pcard.py](code/opus_reexam_20261001_v1/physicist/pcard.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L379](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L379)
- [TEAM_LOG_2026-10-01_to_10-22.md#L387](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L387)
- [RESULT.md#L1611](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L1611)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-block-rewrite-saves-667-requests-but-raises-mse-0) · [Copy provenance](../../../PROVENANCE.json)

