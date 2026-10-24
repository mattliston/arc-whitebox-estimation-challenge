# Dead-row tensor trace restoration: adjusted error rises 0.37808%

2026-09-26 · experiment · rejected · measured

Every actual padded live tensor row was retained and only omitted dead canonical-polyadic tensor diagonals were restored analytically. The compute saving did not offset raw error.

Recorded measurement:

    TRAIN2 raw+1.57060%, bill-1.17600%, adjusted+0.37808%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_dead_tensor_trace_v1.py](code/candidate_dead_tensor_trace_v1.py)

## Evidence

- [RESULTS.md#L627](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L627)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-dead-row-tensor-trace-restoration-adjusted-error-rises-0) · [Copy provenance](../../../PROVENANCE.json)

