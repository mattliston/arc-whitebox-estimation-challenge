# Tail cutoff 2.33 and mean-direction repair: both lose to cutoff 2.5 on six training networks

2026-09-26 · experiment · rejected · measured

A lower fixed cutoff and a conditional mean-direction correction were screened before held-out evaluation. Both were worse than the simpler cutoff-2.5 trace and radial repair and were not validated.

Recorded measurement:

    Cutoff2.33 TRAIN6 adjusted -0.85847%, CI [-1.55428%, -0.22286%], raw +0.50630%; mean-direction repair -0.98600%, CI [-1.56092%, -0.48983%], raw +0.18814%; simpler cutoff2.5 -1.14301%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_old_tail_trace_radial_z233_v3.py](code/candidate_old_tail_trace_radial_z233_v3.py)

## Evidence

- [RESULTS.md#L472](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L472)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-tail-cutoff-2-33-mean-direction-repair-both-lose) · [Copy provenance](../../../PROVENANCE.json)

