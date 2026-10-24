# Covariance-only classical products stabilize layer 9 but final response still fails

2026-09-27 · experiment · rejected · measured

Replacing only the native covariance arithmetic reduced the first-layer secant gap substantially. Remaining jumps at joins10 and13 left the final response outside qualification.

Recorded measurement:

    Post9 secant gaps61–63% ->2.95–2.99%;final gaps45.52/48.33% still fail.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/classical_covariance_response_v1/assess.py](code/codex_resume_20260926_v2/classical_covariance_response_v1/assess.py)
- [code/codex_resume_20260926_v2/classical_covariance_response_v1/build.py](code/codex_resume_20260926_v2/classical_covariance_response_v1/build.py)
- [code/codex_resume_20260926_v2/classical_covariance_response_v1/candidate.py](code/codex_resume_20260926_v2/classical_covariance_response_v1/candidate.py)
- [code/codex_resume_20260926_v2/classical_covariance_response_v1/launch.py](code/codex_resume_20260926_v2/classical_covariance_response_v1/launch.py)
- [code/codex_resume_20260926_v2/classical_covariance_response_v1/run.py](code/codex_resume_20260926_v2/classical_covariance_response_v1/run.py)
- [code/codex_resume_20260926_v2/classical_covariance_response_v1/verify.py](code/codex_resume_20260926_v2/classical_covariance_response_v1/verify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/native_covariance_innovation_fit_v1/candidate.py](code/codex_resume_20260926_v2/native_covariance_innovation_fit_v1/candidate.py)

## Evidence

- [RESULTS.md#L60](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L60)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-covariance-only-classical-products-stabilize-layer-9-but-final) · [Copy provenance](../../../PROVENANCE.json)

