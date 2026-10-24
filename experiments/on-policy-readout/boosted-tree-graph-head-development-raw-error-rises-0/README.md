# Boosted-tree graph head: development raw error rises 0.083935%

2026-09-27 · experiment · rejected · measured

Fixed boosted-tree models used four network folds within128 earlier fitting networks, excluding the prior check32 and validation40. Neither graph nor local-feature control qualified.

Recorded measurement:

    Graph raw+0.083935%,95% interval[-0.042269%,+0.220396%],65/128 wins;local control-0.015929%,interval[-0.152793%,+0.126083%],69/128 wins;all8 exported trees match predictor to1e-12.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/native_final_tree_crossfit_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/assess.py](code/assess.py)
- [code/run.py](code/run.py)
- [code/tree_arrays.py](code/tree_arrays.py)

## Evidence

- [RESULTS.md#L1310](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1310)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-boosted-tree-graph-head-development-raw-error-rises-0) · [Copy provenance](../../../PROVENANCE.json)

