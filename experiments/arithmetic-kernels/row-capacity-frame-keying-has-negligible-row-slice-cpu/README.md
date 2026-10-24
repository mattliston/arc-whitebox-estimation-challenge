# Row-capacity frame keying has negligible row-slice CPU cost and promises 0.66% less warm work

2026-09-24T18:54Z · experiment · informational · measured

Reusing frames keyed by column dimensions while slicing only rows avoids repeated binding. The previous pessimistic compute estimate had been distorted by paging.

Code availability: The retained CPU profiler, warm-frame census and stride_bench.py support this costing study. The reported rows-only rerun (additions 0.91–1.01x) has no separate retained driver; stride_bench.py contains mixed row/column capacity cases, so these attachments do not reproduce that exact rerun.

Recorded measurement:

    - 2026-09-24 11:54 [engineer] WALL PRICING of copy-free levels and less binding (fable_engineer_v1/RESULT.md E-50, nothing built). S11 local user CPU: transform adds 24-26 s, leaf matmuls 11 s, copies 3.6-3.9 s, binding ~0 (views). COLUMN KEYING (bind at row capacity, key on column dims, slice rows) saves -1.15% cold / -0.90% warm (-0.66% 5-net warm mean) and 240-480 requests per net for ~0 s: rows-only slicing costs nothing measurable on user CPU (adds 0.91-1.01x), so E-44's +1.5-5 s estimate (measured while paging) does not hold. Full capacity keying (+24% adds, ~+6-7 s CPU) and row keying are not buyable. Copies: level L-1 elimination in every product = -0.85% for ~+12.6k requests (+3.6-4.1 s wall): no; in the ~60 largest products only ~-0.45% at ~0 s (paper, kernel rewrite ~1 day). Answer: ~1.1-1.6% exact bill looks buyable for ~0 s, not 3-5 s; next = prototype column keying (read frames first).
    - 2026-09-24 12:02 lead: TWO EXACT BILL LEVERS GO.
      (1) Column keying (engineer E-50): bind frames at row capacity, key them on column dims, slice rows. Values are identical, so it is exact. Bill -0.66% warm / -1.15% cold at ~0 s wall (row-only slicing costs no CPU; E-44's +1.5-5 s was measured while paging); requests -240..-480/net. Built as the second bit-exact flag on --colpack (4-6 h).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/op_cpu_profile.py](code/op_cpu_profile.py)
- [code/stride_bench.py](code/stride_bench.py)
- [code/warm_frame_census.py](code/warm_frame_census.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L368](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L368)
- [TEAM_LOG_2026-09-22_to_09-26.md#L370](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L370)
- [TEAM_LOG_2026-09-22_to_09-26.md#L371](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L371)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../grouped-winograd-requests-copied-operands-fewer-requests-but-no/RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-row-capacity-frame-keying-has-negligible-row-slice-cpu) · [Copy provenance](../../../PROVENANCE.json)

