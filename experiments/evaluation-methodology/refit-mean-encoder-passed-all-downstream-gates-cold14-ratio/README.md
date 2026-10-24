# Refit mean encoder passed all downstream gates: cold14 ratio 0.9787744692559065, independent public8 0.9719722866477218

2026-09-21T14:49Z · experiment · adopted · measured

After native40, the unchanged refit candidate passed the 55-call downstream stage: the cold 14-network panel and 8-network confirmation (reused TRAIN/VALID regression checks with own cold bills) and the separate independently sampled 4M-truth panel in which both the public8 and the archived virgin6 groups must improve the unclipped noise-corrected adjusted mean. Both independent groups improved strictly, which admitted the local release.

Recorded measurement:

    cold14 ratio 0.9787744692559065 with 12/14 wins; confirmation8 0.9853436418458571 with 7/8; independent noise-corrected public8 0.9719722866477218, virgin6 0.9899910644606413, combined14 0.9786879737354944; 55 calls, 32 guards, peak RSS 7250165760 bytes, wall 1671.0607975 s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/run.py](code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/native.py](code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/score_stage.py](code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/score_stage.py)
- [code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/seal_stage.py](code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/seal_stage.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/guard.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/common.py](code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/qualify_control.py](code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/qualify_control.py)
- [code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/records.py](code/codex_resume_20260919_v1/current_p_downstream_retirement_v1/records.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)

</details>

## Evidence

- [downstream_release_checkpoint101.md#L3](../../../instruments/downstream-scorer-checks-cold14-confirmation8-independent-public8-plus-virgin6/downstream_release_checkpoint101.md#L3)
- [REEXAMINATION_INDEX_20260919.md#L448](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L448)
- [REEXAMINATION_INDEX_20260919.md#L449](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L449)

<details>
<summary>All result files and records (6)</summary>

- [result.json](result.json) · result
- [panel_score.json](panel_score.json) · result
- [confirmation_score.json](confirmation_score.json) · result
- [independent_score.json](independent_score.json) · result
- [downstream_release_checkpoint101.md](../../../instruments/downstream-scorer-checks-cold14-confirmation8-independent-public8-plus-virgin6/downstream_release_checkpoint101.md) · record
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- measured by: [Downstream scorer checks cold14, confirmation8 and independent public8 plus virgin6 within 55 calls](../../../instruments/downstream-scorer-checks-cold14-confirmation8-independent-public8-plus-virgin6/README.md)
- enables: [Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9](../../../submissions/331760/README.md)
- shipped in: [Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9](../../../submissions/331760/README.md)

[Research graph](../../../graph/index.html#node=2026-09-21-refit-mean-encoder-passed-all-downstream-gates-cold14-ratio) · [Copy provenance](../../../PROVENANCE.json)

