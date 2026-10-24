# Mean-consistency strength and late fourth-order row factors: full-32 tests reject apparent six-network gains

2026-10-20T05:32:18Z · experiment · rejected · measured

Six-network screens suggested a small gain from the mean-consistency coefficient and late fourth-order row-factor scale. Full-32 tests rejected row factors x1.2 and x1.4 and coefficient 0.3; coefficient 0.2 failed three native launches and was not assigned an accuracy verdict.

Recorded measurement:

    - [fable-lead] 2026-10-20T05:32:18Z PRESCREEN 2 (6 store nets, warm-up fixed, MSE): _BFIX_GAMMA 0.119 -> 0.2 MSE -0.39% (4/6, sd 1.8%), -> 0.05 +1.37% (a slope: lane arms bfix_020 / bfix_030 queued); K31R late rows x1.2 -0.14% (3/6), x0.8 +1.10% (lane arms k31rl_12 / k31rl_14 queued); L15 rowcut z 3.5: MSE -0.09% for bill +0.45% DEAD, z 2.5 +4% DEAD; edge / gk4 / rc / iso settings are no-ops in the in-process harness (knob caches) - not retested (FD-scanned before).
    - [fable-lead] 2026-10-20T06:09:00Z PRESCREEN 3 (6 nets): _BFIX_GAMMA 0.25 +0.4% (sd 3%), 0.35 +4.1%, 0.5 +14.6% (0.2 was -0.4%: a noisy optimum near 0.2); K31R late x1.4 +0.55%, x1.6 +2.2%, all-layer x1.2 +1.3%; 0.25 + late x1.2 together -0.2%. Both knobs flat within noise; the 32-rank lanes (bfix_020/030, k31rl_12/14) decide; expectation <= 0.1%.
    - [fable-lead] 2026-10-20T06:14:47Z LANES (32 store ranks vs rel_v4a): k31rl_14 (K31 row factor x1.4 at L10-14) adjusted +1.16% (6/32 lower) DEAD; bfix_030 (_BFIX_GAMMA 0.3) +2.29% DEAD; k31rl_12 and bfix_020 relaunched (their first runs were killed by memory pressure beside the K22 oracle; driver now 2 lanes while oracles run). K22 plane oracle (truth at L10-14): +1.77% / -0.69% (nets 35/100) = net-dependent, no lever.
    [fable-lead] 2026-10-20T08:19:24Z LANE VERDICTS (32 ranks, compare32 vs science32_rel_v4a_m15 = #337259 physics): k31rl_12 (late K31 row factor x1.2 at L10-14) all32 adjusted +0.102% (se 0.123, lower 13/32), raw +0.103%, bill 0 -> DEAD (the x1.4 arm was +1.2%: the live row factor is at its optimum). bfix_020 (_BFIX_GAMMA 0.2) FAILED three launches with native_exit 1 in the census runner (not judged; the 0.05/0.25/0.3/0.35/0.5 arms are all worse, family DEAD regardless). Queue v5 empty, driver exited 06:34Z. Tally for the night stands: no arm at or under -0.1% adjusted except gnnglate_m (-0.055%, 7 ranks); nothing stacks to the 0.005e-9 fire bar.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_bfix_020_sci_v1.py](code/candidate_bfix_020_sci_v1.py)
- [code/candidate_bfix_030_sci_v1.py](code/candidate_bfix_030_sci_v1.py)
- [code/candidate_k31rl_12_sci_v1.py](code/candidate_k31rl_12_sci_v1.py)
- [code/candidate_k31rl_14_sci_v1.py](code/candidate_k31rl_14_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6138](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6138)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6139](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6139)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6141](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6141)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6144](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6144)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-mean-consistency-strength-late-fourth-order-row-factors-full) · [Copy provenance](../../../PROVENANCE.json)

