# Third-order plane sums converted to float32: 0.52% less work with rounding-level error

2026-09-23T08:57Z · experiment · adopted · measured

Using float32 for plane sums reduced the float64 billing share and memory footprint. Corrected quiet measurements showed neutral server compute.

Recorded measurement:

    - 2026-09-23 01:57 [engineer] FLOAT64 CENSUS + F32 PATCH (fable_engineer_v1/dtype_census.py, build_f32.py): S3-HE bills 2.0% of its flops at the float64 rate (max saving 1.0%): third-order planes 0.92%, n x n covariance state 0.55%, batched n x n 0.21%. Patch 'third' (plane sums in float32; pp, lam already float32): util -0.52% r35/r100, float64 share 2.01% -> 1.10%, MSE -0.21/-0.08/+0.12/+0.16% (r35/r100/r37/r39, mean -0.003%), quiet server CPU r35 -0.3..-2.2 s, peak RSS ~-0.15 GB. 'bd' on top: -0.10% more bill, MSE +0.24/-0.25/+0.11/+0.27% (optional). Per-net rounding noise of the final MSE is ~0.2% sd for any rounding-level change, so a +0.05% bound needs VALID40. Stacks as build_eng.py --f32 third (composes with the L15 row cut). Sent to engineer2 for review (his e2 code). Trade menu for depth sent to lead (depth_menu.py).
    - 2026-09-23 02:03 [engineer] CORRECTION to 01:57: f32 'third' is compute-NEUTRAL (clean quiet pair -0.26 s); the first window-4 pair (HE 35.8 vs f32 33.6 s) overlapped engineer2's memory run. Bill/MSE unchanged. Arena answer to engineer2: pure scratch between kernel calls (W5 leaf is a separate copy; frame caches hold views only); the pair map reaches no kernel call.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/dtype_census.py](code/dtype_census.py)
- [code/build_f32.py](code/build_f32.py)
- [code/build_eng.py](code/build_eng.py)
- [code/depth_menu.py](code/depth_menu.py)
- [code/batch_fragment.py](code/batch_fragment.py)
- [code/bigarena_fragment.py](code/bigarena_fragment.py)
- [code/bindhold_fragment.py](code/bindhold_fragment.py)
- [code/caps_fragment.py](code/caps_fragment.py)
- [code/caps_fragment_v3.py](code/caps_fragment_v3.py)
- [code/coldbind_fragment.py](code/coldbind_fragment.py)
- [code/depth_plan_v2.py](code/depth_plan_v2.py)
- [code/evict_fragment_all.py](code/evict_fragment_all.py)

<details>
<summary>Additional programs and dependencies (23)</summary>

- [code/evict_fragment_lru.py](code/evict_fragment_lru.py)
- [code/evict_fragment_push5.py](code/evict_fragment_push5.py)
- [code/evict_fragment_push5r.py](code/evict_fragment_push5r.py)
- [code/grouped_fragment_v5.py](code/grouped_fragment_v5.py)
- [code/grouped_fragment_v7.py](code/grouped_fragment_v7.py)
- [code/grouped_fragment_v7b.py](code/grouped_fragment_v7b.py)
- [code/grouped_fragment_v7c.py](code/grouped_fragment_v7c.py)
- [code/grouped_fragment_v9.py](code/grouped_fragment_v9.py)
- [code/grouped_fragment_v9d.py](code/grouped_fragment_v9d.py)
- [code/grouped_read5_fragment_v5.py](code/grouped_read5_fragment_v5.py)
- [code/grouped_read5_fragment_v7.py](code/grouped_read5_fragment_v7.py)
- [code/grouped_read5_fragment_v7b.py](code/grouped_read5_fragment_v7b.py)
- [code/grouped_read5_fragment_v7c.py](code/grouped_read5_fragment_v7c.py)
- [code/grouped_read5_fragment_v9.py](code/grouped_read5_fragment_v9.py)
- [code/holdelim_fragment.py](code/holdelim_fragment.py)
- [code/holdonly_fragment.py](code/holdonly_fragment.py)
- [code/lean_fragment_v3.py](code/lean_fragment_v3.py)
- [code/push64_fragment.py](code/push64_fragment.py)
- [code/select_round_fragment.py](code/select_round_fragment.py)
- [code/setupbind_fragment_v3.py](code/setupbind_fragment_v3.py)
- [code/sq6_fragment.py](code/sq6_fragment.py)
- [code/symout_fragment_v3.py](code/symout_fragment_v3.py)
- [code/usedslots_fragment.py](code/usedslots_fragment.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L144](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L144)
- [TEAM_LOG_2026-09-22_to_09-26.md#L147](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L147)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #331992: final third-moment row selection, five-level products, and memory reductions](../../../submissions/331992/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-third-order-plane-sums-converted-float32-0-52-less) · [Copy provenance](../../../PROVENANCE.json)

