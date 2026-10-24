# Dead-side cuts, exact column reuse and deeper reads pass the relaxed adjusted-score gate

2026-09-24T19:24Z · experiment · informational · measured

The combined candidate improved adjusted error by 0.995% with neutral raw error and lower CPU time. It was staged after a relaxed gate but superseded by a faster exact kernel before submission.

Recorded measurement:

    2026-09-24 12:24 porter: S12-lite VALID40 vs S11: adjusted -0.666% [CI -0.824, -0.508] 35/40, raw -0.012% (neutral), util -0.657% -> below the fire bar alone (base). S12 smoke (client/server) OK; S12 VALID40 running (~12:40, expect ~-0.98%, at the bar). T640 mid80 VALID16 +0.29% vs B2 ep1 -> no switch.
    2026-09-24 12:25 porter: lead relaxed S12's fire bar to pooled adjusted <= -0.8% (CI excl 0, >= 26/40). fire_chain_S12.sh still checks -1.0 (running; not edited); stage_fire_S12.sh re-evaluates at -0.8 after it and stages fable_fire_v1/fire_candS12/ on PASS. The lead fires after engineer2's S12-vs-S11 wall projection (expected worst <= 100 s).
    2026-09-24 12:32 porter: S12 (S12-lite + read6q) FAILS the fire bar: {"verdict": "FAIL", "pooled_adjusted_pct": -0.995, "ci": [-1.164, -0.824], "wins": 39, "n": 40, "errors": 0, "bar": {"max_pct": -1.0, "ci_excludes_0": true, "min_wins": 26}, "clears_3e9_bar_minus_1p6"
    2026-09-24 12:33 porter: S12 PASSES the relaxed bar (-0.8%); fire copy staged: fable_fire_v1/fire_candS12/candS12_S11_cp_dc35_r6q_mem.py (FIRE_CHECKS.txt).
    - 2026-09-24 12:48 lead: S12 (S11 + colpack + deadcut35 + read6q, c20d67c7) VALID40 vs S11: adjusted -0.995% [CI -1.164, -0.824] 39/40, raw +0.022%, util -1.022%. It passes the relaxed -0.8% bar but is SUPERSEDED: S13 = S12 + --ipt v3 (engineer; candS12ipt3_mem.py 2a5dd324) is byte-identical in predictions to S12 on r35/r100 (in-process and client/server), with flops -1.75%, user CPU -35%, server wall -28%, live peak 2555->2424 MB, handles 25.7k->16.2k, requests -9k/net. Expected ~2.963e-9, UNDER 3e-9. The porter runs S13's VALID40 (identity vs S12 rows on all 40 nets + bar), stages fire_candS13, and the lead fires.
    2026-09-24 13:50 [engineer2] S12 WALL MEASURED (pair finished 12:37; its notice reached me 13:50): stack_S11_cp_dc35_r6q_mem.py c20d67c7 vs fired S11, porter's pair method, 6 nets concurrent, 1 thread: user CPU -9.17% (se 0.65%; per net -6.8..-11.4%), system flat (2.9-3.3 s both arms), flops -0.96%, op calls +668/net. Expected worst (telemetry/project_wall_v2.py on S11 97.6 s / kernel 84.91 s; user ratio 0.57/0.75/1.0): 93.3 / 91.9 / 90.0 s, pessimistic room 6.7 s. S13 (= S12 + ipt v3) needs its own pair (kernel change).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/fire_chain_S12.sh](code/fable_porter_v1/fire_chain_S12.sh)
- [code/fable_porter_v1/stage_fire_S12.sh](code/fable_porter_v1/stage_fire_S12.sh)
- [code/fable_fire_v1/fire_candS12/candS12_S11_cp_dc35_r6q_mem.py](code/fable_fire_v1/fire_candS12/candS12_S11_cp_dc35_r6q_mem.py)
- [code/fable_engineer_v1/candS12ipt3_mem.py](code/fable_engineer_v1/candS12ipt3_mem.py)
- [code/fable_porter_v1/stack/stack_S11_cp_dc35_r6q_mem.py](code/fable_porter_v1/stack/stack_S11_cp_dc35_r6q_mem.py)
- [code/fable_engineer2_v1/telemetry/project_wall_v2.py](code/fable_engineer2_v1/telemetry/project_wall_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L384](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L384)
- [TEAM_LOG_2026-09-22_to_09-26.md#L385](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L385)
- [TEAM_LOG_2026-09-22_to_09-26.md#L386](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L386)
- [TEAM_LOG_2026-09-22_to_09-26.md#L389](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L389)
- [TEAM_LOG_2026-09-22_to_09-26.md#L393](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L393)
- [TEAM_LOG_2026-09-22_to_09-26.md#L400](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L400)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-dead-side-cuts-exact-column-reuse-deeper-reads-pass) · [Copy provenance](../../../PROVENANCE.json)

