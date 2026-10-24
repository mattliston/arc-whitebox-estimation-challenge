# Seeded Gaussian star range restored: adjusted error rises 0.48684% despite negligible raw change

2026-09-27 · experiment · rejected · measured

The older seeded Gaussian range finder replaced the deployed Nyström range while preserving48 columns, one power step, rank32 truncation and all other native code. This isolated range mismatch failed as a direct candidate and does not bound matched retraining.

Recorded measurement:

    TRAIN8 raw-0.01225%,adjusted+0.48684%,paired95% interval[+0.01422%,+1.00505%],2/8 adjusted wins;bill+0.50356%;prediction difference0.24009% of parent residual squared energy,4.89989% of RMS,residual cosine-0.025749.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyze_native_training_star_counterfactual_v1.py](code/analyze_native_training_star_counterfactual_v1.py)
- [code/build_native_training_star_counterfactual_v1.py](code/build_native_training_star_counterfactual_v1.py)
- [code/candidate_training_star_counterfactual_v1.py](code/candidate_training_star_counterfactual_v1.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)

## Evidence

- [RESULTS.md#L1053](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1053)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-seeded-gaussian-star-range-restored-adjusted-error-rises-0) · [Copy provenance](../../../PROVENANCE.json)

