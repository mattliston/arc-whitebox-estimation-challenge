# Two-tail old-row reconstruction: all 40 held-out adjusted scores improve by a pooled 2.40106%

2026-09-26 · experiment · adopted · measured

Only old rows below -2.5 that were also absent from the actual live roster were removed, in addition to the positive-tail cut. Padded live negative rows, all young columns and the full carrier remained, with analytic diagonal and radial restoration.

Recorded measurement:

    TRAIN6 adjusted -2.26833%, CI [-3.16702%, -1.49679%],6/6 wins; fresh VALID10 -2.35311%, CI [-2.77293%, -1.94585%],10/10 wins; VALID40 -2.40106%, CI [-2.65828%, -2.15236%],40/40 wins; raw +0.26756%, CI [+0.02458%, +0.50796%]; compute -2.65846%; CPU -3.04218%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_z25_v5.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_z25_v5.py)
- [code/codex_resume_20260926_v1/candidate_bundle_diagreuse_v1.py](code/codex_resume_20260926_v1/candidate_bundle_diagreuse_v1.py)

## Evidence

- [RESULTS.md#L536](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L536)
- [RESULTS.md#L513](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L513)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [both_tail_valid40_comparison_v5.json](both_tail_valid40_comparison_v5.json) · result

</details>

## Related work

- shipped in: [Submission #332517: two-tail old-row reconstruction with analytic diagonal and radial restoration](../../../submissions/332517/README.md)

[Research graph](../../../graph/index.html#node=2026-09-26-two-tail-old-row-reconstruction-all-40-held-out) · [Copy provenance](../../../PROVENANCE.json)

