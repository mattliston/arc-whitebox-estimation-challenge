# In-place tree kernel reduces platform predict wall 27.48 seconds and recalibrates the timing model

2026-09-24T21:07Z · finding · informational · measured

Paired official telemetry confirmed most of the local user-CPU gain. The wall projection tool was updated to apply a measured ratio band to the worst-network kernel time.

Recorded measurement:

    2026-09-24 14:07 [engineer2] S13 WALL MEASURED (for the record; base for the engineer's S14 depth re-optimization): candS12ipt3_mem.py 2a5dd324 vs fired S11, pair method (concurrent, 1 thread, 6 nets 35c/100/37/39/42/48, run 13:52-13:58): user CPU -40.2% (se 0.75%; per net -37.5..-42.3%), system flat (4.2-5.5 s both arms), flops -2.69%, op calls -1,732/net. Expected worst (project_wall_v2 on S11 97.6 s / kernel 84.91 s; user ratio 0.57/0.75/1.0 = pess/central/opt): 77.8 / 71.6 / 63.1 s, room below 100 s 22.2 / 28.4 / 36.9 s (vs S12 93.3 / 91.9 / 90.0 s). CAVEAT: the 0.57-1.0 ratio band was calibrated on ~5% kernel changes; -40% is far outside it, so S13's own platform telemetry should confirm before S14 spends more than ~15 s of it. Housekeeping: my first S13 pair (fast-poll runner) had actually completed when I stopped it, but my kill left /tmp/fable_paired.lock held from 13:51:52; the porter released it ~14:06; my redundant post-fire rerun (s13_wall_after_fire.sh) is cancelled.
    - 2026-09-24 14:32 lead: GRADED #332195 (S13) = 2.9743e-9, NEW BANK, FIRST SUB-3e-9 (-2.37% vs S11). Raw 1.7739e-8 (+0.17% vs S11 on the public 50, noise), util .16776. 50/50 nets better adjusted. PLATFORM WALL: roundtrip mean 66.2 s / max 73.2 s (S11 93.6/99.3), kernel 158 ns/kflop (S11 226), 0 failed, residual max 0.28 s. That leaves ~27 s of measured room below 100 s for S14 (engineer2's pessimistic 77.8 s was conservative). Next milestone 2e-9 needs -32.8%.
    2026-09-24 14:22 [engineer2] WALL MODEL RECALIBRATED on S13 (#332195 vs #332182, 100 MLPs paired): platform kernel -31.53% (-26.83 s), overhead -0.64 s, comms -0.07 s, predict wall -27.48 s, expected worst 97.6 -> 68.9 s; my local user-CPU pair said -40.20%, op calls -1,732/net. => r = platform kernel % / local user % = 0.784 (push5p6's two platform runs gave 0.57 / 0.94). New tool fable_engineer2_v1/telemetry/project_wall_v3.py: r band 0.60 / 0.78 / 0.95; the % change is applied to the WORST MLP's kernel (K_mean x fit_max / predict_mean of the base page; S11 88.5 s vs mean 84.9) because the expected worst is the heaviest MLP's fitted wall; op calls x 0.41 ms (0.30-0.50; S13: -0.71 s overhead+comms for -1,732 op calls); scenarios ordered by the resulting wall (v2 mislabelled pess/opt for CPU ADDITIONS). Backtest from S11 + my S13 pair: 75.7 / 69.1 / 62.9 s vs actual 68.9 s. FOR S14 (base #332195: expected worst 68.9 s, kernel 58.08 s mean, 60.50 s worst MLP): +1% paired local user CPU vs S13 = +0.47 s expected worst (0.36-0.58); +1,000 op calls = +0.41 s. Room below 100 s = 31.1 s, i.e. about +52% local user CPU at the pessimistic rate (+66% central). Usage: python3 project_wall_v3.py --base=332195 --pair=NAME:optime_cand.json:optime_S13ref.json (pairs vs S13 via the porter's runner).

## Programs

- [../../../experiments/evaluation-methodology/dead-side-cuts-exact-column-reuse-deeper-reads-pass/code/fable_engineer_v1/candS12ipt3_mem.py](../../../experiments/evaluation-methodology/dead-side-cuts-exact-column-reuse-deeper-reads-pass/code/fable_engineer_v1/candS12ipt3_mem.py)
- [s13_wall_after_fire.sh](s13_wall_after_fire.sh)
- [project_wall_v3.py](project_wall_v3.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L403](../../TEAM_LOG_2026-09-22_to_09-26.md#L403)
- [TEAM_LOG_2026-09-22_to_09-26.md#L406](../../TEAM_LOG_2026-09-22_to_09-26.md#L406)
- [TEAM_LOG_2026-09-22_to_09-26.md#L407](../../TEAM_LOG_2026-09-22_to_09-26.md#L407)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-place-tree-kernel-reduces-platform-predict-wall-27-48) · [Copy provenance](../../../PROVENANCE.json)

