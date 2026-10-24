# Magnitude-aware old-row allocation: native adjusted error rises 41.703189%

2026-09-27 · experiment · rejected · measured

Gaussian curvature and normalized row magnitude ranked old-row reconstructions, retaining128 fewer rows per call while protecting live negative rows. The complete allocation rule preserved the carried tensor, young columns and repairs but failed decisively.

Recorded measurement:

    TRAIN35/100 raw+43.504640%,bill-1.256551%,adjusted+41.703189%,0/2 wins;individual adjusted+44.123438%/+39.436417%;8 calls omit1024 additional row reconstructions/network.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/old_row_importance_v1/assess.py](code/old_row_importance_v1/assess.py)
- [code/old_row_importance_v1/build_and_qualify.py](code/old_row_importance_v1/build_and_qualify.py)
- [code/old_row_importance_v1/candidate.py](code/old_row_importance_v1/candidate.py)
- [code/old_row_importance_v1/close_checkpoint27.py](code/old_row_importance_v1/close_checkpoint27.py)
- [code/old_row_importance_v1/evaluate.py](code/old_row_importance_v1/evaluate.py)
- [code/old_row_importance_v1/fragment.py](code/old_row_importance_v1/fragment.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [RESULTS.md#L1268](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1268)

<details>
<summary>All result files and records (3)</summary>

- [RESULTS.md](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result
- [assessment.json](assessment.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-magnitude-aware-old-row-allocation-native-adjusted-error-rises) · [Copy provenance](../../../PROVENANCE.json)

