# Monte Carlo truth chain: 64M-sample final-layer truths for fresh networks, noise 1.19e-9, 2,200-2,700 cached

2026-10-14T01:39Z · tool · informational · measured

The analyst's truth chain (mc_truth_v2.py, bit-identical to mc_truth.py) kept producing final-layer Monte Carlo truths for fresh networks in rounds ordered by need (16M or 8M samples per round, 536 s of GPU per network), feeding the refit rounds and the fresh-seed screens. The final-layer truth noise of the 64M-sample truths is 1.19e-9 mean-squared.

Recorded measurement:

    2,188 truths ready at 01:39Z, 2,700 at 15:21Z (seeds 7100000-7190063); 536 s GPU per network over the last 50; final-layer truth noise 1.19e-9 (mean-squared, 64M samples); rounds A76 (S57 reserve, N=15,990,784), A78/A80 (S58 phase 1, N=7,995,392), A79 (S58 check, 16M), A82, A84/A86 (S59 phase 1, 8M), A85 (S59 check, 16M), A90 (S60 phase 1, 8M)

## Programs

- [code/mc_truth_v2.py](code/mc_truth_v2.py)
- [code/loader.py](code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2492](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2492)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2528](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2528)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2614](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2614)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2646](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2646)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2711](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2711)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2722](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2722)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2784](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2784)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2842](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2842)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2860](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2860)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2862](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2862)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2868](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2868)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2899](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2899)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-14-monte-carlo-truth-chain-64m-sample-final-layer-truths) · [Copy provenance](../../PROVENANCE.json)

