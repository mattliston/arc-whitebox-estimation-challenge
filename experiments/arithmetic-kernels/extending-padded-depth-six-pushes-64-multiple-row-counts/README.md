# Extending padded depth-six pushes to 64-multiple row counts costs 49-93% more user CPU

2026-09-24T19:23Z · experiment · rejected · measured

The extension recovered work lost by smaller row counts but spent too much runtime for the gain. It was not recommended at the current wall budget.

Recorded measurement:

    - 2026-09-24 12:23 [engineer] --m64 BUILT AND MEASURED, NOT RECOMMENDED (fable_engineer_v1/RESULT.md E-51): six-level pushes for m % 64 rows (post-stage build_m64.py, bit-exact at m = 1024: null build identical to colpack). Row bench: current depth-5 fallback bills +5-6% per row but costs 455-637 ms; six levels recover the bill (-4..-5%) at +49..+93% user CPU. Whole net vs S11: deadcut-3.5 r64 -0.48/-0.95% -> +m64 -1.06/-1.55% (r35/r100), at +3.4..+4.6 s local user CPU (+7-10%, A-B-A-B in process) = ~0.1% of bill per platform-second (read6q ~0.15%/s, column keying ~free). -3.0: -1.29/-1.53% (r32), -1.86/-2.04% with m64; its extra ~0.7% bill roughly cancels the analyst's +0.88% raw. Keep deadcut-3.5 on current kernels (its -10% CPU funds read6q). Porter's 6-net pair of m64 vs dc35r64 queued behind the VALID40 lock.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_m64.py](code/build_m64.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L383](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L383)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../grouped-winograd-requests-copied-operands-fewer-requests-but-no/RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-extending-padded-depth-six-pushes-64-multiple-row-counts) · [Copy provenance](../../../PROVENANCE.json)

