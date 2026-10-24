# Frozen covariance-edge correction passes panel14 at ratio 0.982838708008 and confirmation8 at 0.982181085851

2026-09-20 · experiment · adopted · measured

The fixed amplitude-one pair-local covariance correction at layers8–14 passed its separately declared autonomous final-score study without refitting or a layer sweep. Its old strict covariance and response proxy failures remain recorded; the later lossless packed derivative was submitted as#331684.

Recorded measurement:

    Development ratio0.982783220745;panel14 0.982838708008,12/14 wins;confirmation8 0.982181085851,6/8 wins,worst1.025721997737;29full estimator calls,31guards,peak6907297792B,zero added swap.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/full_lower_covariance_edge_native_v1/build.py](code/full_lower_covariance_edge_native_v1/build.py)
- [code/full_lower_covariance_edge_native_v1/candidate.py](code/full_lower_covariance_edge_native_v1/candidate.py)
- [code/full_lower_covariance_edge_native_v1/common.py](code/full_lower_covariance_edge_native_v1/common.py)
- [code/full_lower_covariance_edge_native_v1/edge_helper.py](code/full_lower_covariance_edge_native_v1/edge_helper.py)
- [code/full_lower_covariance_edge_native_v1/finalize.py](code/full_lower_covariance_edge_native_v1/finalize.py)
- [code/full_lower_covariance_edge_native_v1/native.py](code/full_lower_covariance_edge_native_v1/native.py)
- [code/full_lower_covariance_edge_native_v1/panel_run.py](code/full_lower_covariance_edge_native_v1/panel_run.py)
- [code/full_lower_covariance_edge_native_v1/panel_score.py](code/full_lower_covariance_edge_native_v1/panel_score.py)
- [code/full_lower_covariance_edge_native_v1/qualify.py](code/full_lower_covariance_edge_native_v1/qualify.py)
- [code/full_lower_covariance_edge_native_v1/reference.py](code/full_lower_covariance_edge_native_v1/reference.py)
- [code/full_lower_covariance_edge_native_v1/run_development.py](code/full_lower_covariance_edge_native_v1/run_development.py)
- [code/full_lower_covariance_edge_native_v1/score.py](code/full_lower_covariance_edge_native_v1/score.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/full_lower_covariance_edge_native_v1/unit.py](code/full_lower_covariance_edge_native_v1/unit.py)
- [code/full_lower_moment_panel_v1/candidate.py](code/full_lower_moment_panel_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [ACCURACY_RESULT.md#L4](ACCURACY_RESULT.md#L4)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [ACCURACY_RESULT.md](ACCURACY_RESULT.md) · record

</details>

## Related work

- ← measured by: [Submission #331684: packed frozen covariance-edge correction at layers 8-14 with full lower-moment support](../../../submissions/331684/README.md)

[Research graph](../../../graph/index.html#node=2026-09-20-frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008) · [Copy provenance](../../../PROVENANCE.json)

