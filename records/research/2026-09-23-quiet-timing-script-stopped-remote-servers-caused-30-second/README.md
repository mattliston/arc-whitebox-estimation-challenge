# Quiet timing script stopped remote servers and caused 30-second client timeouts

2026-09-23T06:14Z · incident · failed · measured

The first quiet-timing script paused other flopscope servers as well as computation jobs. The replacement excluded processes with ZeroMQ loaded.

Recorded measurement:

    2026-09-22 23:14 [engineer2] fire gates: candA memory PLATEAU over 10 distinct nets (peak 2.71-2.77 GB live, +665 MB vs bank, live 1.61 GB after each), handles track the bank (53.7k vs 50.8k after 10 nets; 100k store limit). Timing A/B (load 10-13): candA server CPU +17%/+22% vs bank, e2+K31 -9%/-1%. quiet_ab.sh ready. Quiet-window SIGSTOP kills other lanes' flopscope runs (client 30 s request timeout).
    - 2026-09-22 23:24 lead: RULE: use fable_fire_v1/quiet_census_v2.sh from now on (v1 kept). v1 SIGSTOPped other lanes' flopscope servers, and the client aborts any request after 30 s, so it killed engineer2's c4 memory run at r39. v2 pauses only heavy python jobs WITHOUT the ZeroMQ library loaded; flopscope servers and clients keep running and are logged. S3 BASE = fable_engineer_v1/candS3_S2_HE.py (S2 + --holdelim, no bigarena, bit-identical to S2). Engineer's 23:09 quiet census vs bank, r35/r100: wall 34.9/31.8 vs 43.6/39.0 s, server CPU 31.5/28.9 vs 38.1/34.1 s, round trips 53.8k/46.0k vs 95.4k/84.1k, peak RSS 5.78 vs 6.31 GB -> wall-safe. Porter HEB VALID40 adjusted 0.93406 (HE expected ~0.937). S3 is a base, not a fire; the next fire stacks S3 + knobs + refit + sq6 if together >= 3% over S2.

## Programs

- [quiet_ab.sh](quiet_ab.sh)
- [quiet_census_v2.sh](quiet_census_v2.sh)
- [../../../experiments/arithmetic-kernels/held-aliases-last-level-copy-elimination-preserve-predictions-reduce/code/fable_engineer_v1/candS3_S2_HE.py](../../../experiments/arithmetic-kernels/held-aliases-last-level-copy-elimination-preserve-predictions-reduce/code/fable_engineer_v1/candS3_S2_HE.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L83](../../TEAM_LOG_2026-09-22_to_09-26.md#L83)
- [TEAM_LOG_2026-09-22_to_09-26.md#L86](../../TEAM_LOG_2026-09-22_to_09-26.md#L86)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-quiet-timing-script-stopped-remote-servers-caused-30-second) · [Copy provenance](../../../PROVENANCE.json)

