# Depth-1 third-cumulant arena deployed in submission #328586

2026-08-26 · experiment · adopted · measured

The depth-1 arena preserves output to float32 roundoff while saving about 118.5B hot third-cumulant canonical-polyadic carrier (CP3) FLOPs. Its local A/B/A gate failed only the final rank35 residual by 0.00020734211429954 s, but the official grader is authoritative: submission #328586 graded 100/100 at adjusted 2.1258043062525995e-8 and raw 2.9409214512554626e-8, improving #328439 adjusted by 6.94% with raw effectively unchanged.

Recorded measurement:

    The depth-1 arena preserves output to float32 roundoff while saving about 118.5B hot CP3 FLOPs. Its local A/B/A gate failed only the final rank35 residual by 0.00020734211429954 s, but the official grader is authoritative: submission #328586 graded 100/100 at adjusted 2.1258043062525995e-8 and raw 2.9409214512554626e-8, improving #328439 adjusted by 6.94% with raw effectively unchanged.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)

## Evidence

- [D1_CANDIDATE_RESULT.json#L1](D1_CANDIDATE_RESULT.json#L1)
- [SUBMIT_WATCH.log#L1](../../flop-accounting/deployed-estimator-score-economics-minimum-charged-utilisation-verified/SUBMIT_WATCH.log#L1)

<details>
<summary>All result files and records (2)</summary>

- [D1_CANDIDATE_RESULT.json](D1_CANDIDATE_RESULT.json) · result
- [SUBMIT_WATCH.log](../../flop-accounting/deployed-estimator-score-economics-minimum-charged-utilisation-verified/SUBMIT_WATCH.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-depth-1-third-cumulant-arena-deployed-submission-328586) · [Copy provenance](../../../PROVENANCE.json)

