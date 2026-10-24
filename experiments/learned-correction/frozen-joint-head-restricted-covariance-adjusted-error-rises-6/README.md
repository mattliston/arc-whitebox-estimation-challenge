# Frozen joint head restricted to covariance: adjusted error rises 6.99703%

2026-09-27 · experiment · rejected · measured

The same frozen joint-head predictions were applied only to covariance. Restricting the consumer reduced the severe full-transfer loss but still failed both training networks.

Recorded measurement:

    TRAIN35/100 raw+3.47638%,bill+3.40112%,adjusted+6.99703%,0/2 wins;disabled all16 bytes and full bill exact.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/assess_endpoint.py](code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/assess_endpoint.py)
- [code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/build.py](code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/build.py)
- [code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/candidate.py](code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/candidate.py)
- [code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/delta_fragment.py](code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/delta_fragment.py)
- [code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/disabled.py](code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/disabled.py)
- [code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/unit.py](code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/unit.py)
- [code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/unit_v2.py](code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/unit_v2.py)
- [code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/unit_v3.py](code/codex_resume_20260926_v2/joint_moment_covariance_only_v1/unit_v3.py)
- [code/codex_resume_20260919_v1/joint_incoming_moment_head_v1/head.py](code/codex_resume_20260919_v1/joint_incoming_moment_head_v1/head.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)

## Evidence

- [RESULTS.md#L297](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L297)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-frozen-joint-head-restricted-covariance-adjusted-error-rises-6) · [Copy provenance](../../../PROVENANCE.json)

