# First-use view binding consumes 1.4-1.7% of average metered work

2026-09-23T09:50Z · finding · informational · measured

Reshape views incur metered work when newly bound even without data movement. Cold network costs recur when shapes change, and repeat networks avoid them.

Recorded measurement:

    2026-09-23 02:50 [engineer2] FIRST-TIME VIEW BINDING IS ~1.4-1.7% OF THE AVERAGE BILL (fable_engineer2_v1/bind_bill.py, S5 draft in-process, 10 distinct nets back to back like one worker, then r35 again): share of each net's bill spent in reshape = view binding (all in _exact_view + _read5_view + first-call _cp3_bind_arena_views): r35 cold 4.94%, then r100 2.04, r37 0.79, r39 2.05, r42 1.45, r48 0.54, r59 1.17, r63 1.61, r66 1.77, r67 0.95 (mean 1.37% on nets 2-10, no clear decline), r35 again 0.00. Platform S4 telemetry agrees: MLPs 0-4 (first per worker) bill 0.183-0.188 vs the 0.1785 mean. flopscope bills reshape numel(input) even for views; basic slicing, unstack, split, expand_dims are free. matmul on the server's OpenBLAS is bit-identical with row-padded (strided) operands, outputs and batched items (lda test on 7 shapes up to 1024^3), so exact-shape views could be basic-slice sub-views of the once-bound max-shape region views: ~-1.4% bill bit-exact if the kernel's tail reuse allows it (engineer's call).

## Programs

- [bind_bill.py](bind_bill.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L162](../../TEAM_LOG_2026-09-22_to_09-26.md#L162)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-first-use-view-binding-consumes-1-4-1-7) · [Copy provenance](../../../PROVENANCE.json)

