# Bounded frame caches preserve all 40 predictions and the 2.40010% adjusted gain

2026-09-26 · experiment · adopted · measured

Frame-cache limits changed from 480/480/480 to 320/352/320 without changing formulas, trained parameters or setup keys. Corrected long-worker tests and repeated native validation qualified the exact source for submission.

Recorded measurement:

    VALID40 versus #332492 adjusted -2.40010%, paired95% CI [-2.65752%, -2.15088%],40/40 wins; raw +0.26756%; compute -2.65743%; CPU -3.46455%; all40 predictions bit-identical to unbounded-cache version.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)

## Evidence

- [RESULTS.md#L570](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L570)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

## Related work

- shipped in: [Submission #332517: two-tail old-row reconstruction with analytic diagonal and radial restoration](../../../submissions/332517/README.md)

[Research graph](../../../graph/index.html#node=2026-09-26-bounded-frame-caches-preserve-all-40-predictions-2-40010) · [Copy provenance](../../../PROVENANCE.json)

