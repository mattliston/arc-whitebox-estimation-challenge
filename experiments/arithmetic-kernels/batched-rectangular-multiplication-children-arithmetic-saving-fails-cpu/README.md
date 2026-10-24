# Batched rectangular multiplication children: arithmetic saving fails CPU admission

2026-09-26T23:45Z · experiment · rejected · measured

Two versions batched the same15 rectangular multiplication children within fixed scratch regions, with the second using bounded view caches. They matched sequential outputs exactly but both failed the declared speed gates.

Recorded measurement:

    Version1 CPU0.921642x sequential/2.203212x parent; version2 0.888042x/2.056196x; version2 primitive bill-4.922132%; all54 scalar cases pass; all12 real-shape outputs per version byte-identical; FP64 fixture error7.11e-15.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/bench_hk233_batch_v1.py](code/codex_resume_20260926_v2/bench_hk233_batch_v1.py)
- [code/codex_resume_20260926_v2/bench_hk233_batch_v2.py](code/codex_resume_20260926_v2/bench_hk233_batch_v2.py)
- [code/codex_resume_20260926_v2/build_hk233_batch_v1.py](code/codex_resume_20260926_v2/build_hk233_batch_v1.py)
- [code/codex_resume_20260926_v2/build_hk233_batch_v2.py](code/codex_resume_20260926_v2/build_hk233_batch_v2.py)
- [code/codex_resume_20260926_v2/hk233_batch_fragment_v1.py](code/codex_resume_20260926_v2/hk233_batch_fragment_v1.py)
- [code/codex_resume_20260926_v2/hk233_batch_fragment_v2.py](code/codex_resume_20260926_v2/hk233_batch_fragment_v2.py)
- [code/codex_resume_20260926_v1/hk233_fragment_v1.py](code/codex_resume_20260926_v1/hk233_fragment_v1.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)

## Evidence

- [RESULTS.md#L844](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L844)

<details>
<summary>All result files and records (4)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [hk233_batch_closeout_v1.json](hk233_batch_closeout_v1.json) · result
- [hk233_batch_bench_v1.json](hk233_batch_bench_v1.json) · result
- [hk233_batch_bench_v2.json](hk233_batch_bench_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-batched-rectangular-multiplication-children-arithmetic-saving-fails-cpu) · [Copy provenance](../../../PROVENANCE.json)

