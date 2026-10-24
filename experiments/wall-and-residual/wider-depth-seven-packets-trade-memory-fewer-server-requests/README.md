# Wider depth-seven packets trade memory for fewer server requests

2026-09-25T20:00Z · experiment · inconclusive · measured

Doubling the packet width from 2048 to 4096 leaves the arithmetic bill unchanged and removes 272 requests per network while increasing peak memory by 331 MB. The projected wall saving is not a completed end-to-end gate.

Recorded measurement:

    - 2026-09-25 13:00 [engineer2] OPTION FOR S30 (not S29): 4096-wide depth-7 colpack packets instead of 2048 (depth7/bench_pk.py on S28, arena grown 391.6M -> 474.4M floats, +331 MB RSS since every net writes these trees). Same bill to the flop; ops -68 per 4096 columns (-272/net); backend under load -0.16 s (n=4096), ~0 (6144, noise), -0.59 s (8192, all 5 reps apart). Scaled to quiet-machine rates that is about -0.5 s local = about -1.1 s on the slowest net with the ops, i.e. ~15% of d7cp + d7sq's priced +7.1 s. Needs its own parity (rounding changes: different leaf blocking) and a memory check against the 8 GB rule (server peak 3.5-4.9 GB in the S28 census).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/depth7`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/bench_pk.py](code/bench_pk.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1243](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1243)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-wider-depth-seven-packets-trade-memory-fewer-server-requests) · [Copy provenance](../../../PROVENANCE.json)

