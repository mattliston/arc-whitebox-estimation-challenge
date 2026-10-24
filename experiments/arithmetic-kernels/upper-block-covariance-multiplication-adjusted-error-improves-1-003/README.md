# Upper-block covariance multiplication: adjusted error improves 1.003% but adds requests and residual

2026-09-23T15:04Z · experiment · inconclusive · measured

Computing the upper covariance blocks and mirroring the lower block avoided the failed inferred-symmetry path. The candidate passed accuracy validation but its request and runtime costs required additional reductions.

Recorded measurement:

    - 2026-09-23 08:04 [engineer] REQUESTS/RESIDUAL + WARM-BINDING PRICING (no builds). (a) symout v2's +1,642 requests (engineer2's census, cold r35): the covariance's two standalone kernel calls +2,450, the P=2 batch -420, the old-core einsum about -300 (read5 path). A batched 3-block covariance (v3) would recover only about 400 of them (an extra batched product still costs about 30 requests vs 87 standalone) at +0.05% bill vs v2: not built. The client's residual is spread over flopscope's per-request machinery (client_profile.py: cb3 and v2 look alike per function; client CPU 11.0 vs 11.2 s), i.e. about 14 us per request under load 17 (cb3's own residual reads 0.548 s here vs about 0.18 s in engineer2's quieter runs), so the quiet delta is likely about +7-8 ms; engineer2's gate measures it. (b) The covariance block shapes take depth 5 under both S6b's and S7-full's (push5p6) rule: symout composes. (c) Warm binding (warm_frame_census.py, symout v2, nets 2-6 of one process): 1.29% of the warm bill on average (0.62-1.62% per net): read5 children 0.27%, READ A 0.26%, READ B 0.20%, push O/B frames 0.46%, squares 0.07%. Column keying (bind at full rows, slice rows) has a ceiling of 0.83% (frames whose column count was already bound earlier in the process), realistically 0.5-0.7%, and row-sliced frames benchmarked 4-100% slower per op (B-side transforms are about 40% of push transform time): estimated +1.5 to +5 s CPU per net and 4-6 h of work. Fails the >=0.8% for <=+1 s bar; not recommended.
    2026-09-23 08:39 [engineer2] S8 (symout v3) gate STOPPED per the lead (S8 on hold) after 2 paired nets vs S7 (record: fable_engineer2_v1/g20S8so3.out, g20refS8so3.out; load 17-19): server CPU +1.39% (r35 cold) / +1.42% (r100), round trips +2.6k/+2.8k, residual +25/+47 ms (loaded), util -1.02%/-0.97%. Per-flop check of my paired 20-net gates vs the lead's platform ns/kflop: S6b vs S5e local -2.07% vs platform -1.4% (agree: --batch does not under-read); S7 vs S6b local +3.53% vs platform +8% (the only gap). Of S7's components only push5p6 changes warm kernel compute, so the gap is push5p6 or the host; S9 (same kernel code as S7) will tell: ~232 ns/kflop again = push5p6, ~215 = host. S9 light gate scripted (fable_engineer2_v1/s9/gate_s9.sh: in-process 4 nets with bill identity vs S7 + paired 6 nets), waiting for the porter's file. Calibration: in-process op-class timing of S2/S4/S5e/S6b/S7/S8 running (fable_engineer2_v1/platfit/).
    2026-09-23 08:49 porter: S8 v3 (5116cee9) VALID40 vs S7 rows: raw +0.018% [CI -0.035, +0.070] PASS; util -1.022%; adjusted -1.003% [-1.056, -0.951], 40/40, 0 errors. Parity warm nets -0.92..-1.15% adjusted (the cut holds warm). Vs S6b -3.145%. Pending engineer2's CPU gate.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/client_profile.py](code/fable_engineer_v1/client_profile.py)
- [code/fable_engineer_v1/warm_frame_census.py](code/fable_engineer_v1/warm_frame_census.py)
- [code/fable_engineer2_v1/s9/gate_s9.sh](code/fable_engineer2_v1/s9/gate_s9.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L260](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L260)
- [TEAM_LOG_2026-09-22_to_09-26.md#L276](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L276)
- [TEAM_LOG_2026-09-22_to_09-26.md#L278](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L278)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-upper-block-covariance-multiplication-adjusted-error-improves-1-003) · [Copy provenance](../../../PROVENANCE.json)

