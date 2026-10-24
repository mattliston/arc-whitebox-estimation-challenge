# Score-versus-runtime-risk comparison: a projected 1.0% score gain retains a 46–57% public-failure probability

2026-10-16T18:02:05Z · finding · informational · measured

For the user decision on the last two Oct 16 slots the release lane built the frontier: the clean S3pb (about 1.4757e-9, P(fail) 0.4%), V-C (the estimator submitted as #336412 (S2tr) + light diet: 1.4670e-9 but warm residual median about .348), V-A (S2tr + dm2 + lazybind v2 + the four value-only parts: 1.4606e-9 at S2tr request counts) and V-A+B (adds option B). The repaired iffy builds (V-A+B + byte-identical diets + QL off + own-census prebind refit + multi-slice editor) cut requests by about 2,000 at +0.017% bill but still carried P(public fail) 23-26% and failed the lead confidence bar on residual, probability and wall; no build passed all of (a) projection <= 1.4635e-9 with store-pooled parts, (b) per-net instruction residual <= .325, (c) P(any) <= 3%, (d) worst wall <= 106 s together. Wall is set by the FLOPs, so request diets do not move it.

Recorded measurement:

    S3pb ~1.4757e-9 (median .3157, P family 0.4/0.4%); V-C bill -0.217% vs S2tr -> 1.4670e-9, slow 44,973/43,909, worst 104.1 s, median .3556 instr, P family 71/49%; V-A bill -0.653% -> 1.4606e-9, slow 47,817/45,557, worst 109.5 s, median .3605, P family 80/57%; V-A+B prereg 1.4568e-9 boundary / 1.4585e-9 4-net, median .3555, P family 71/49%. VQpbm 6ecac031: normal 45,364/43,709, slow 45,762/44,055, median .3417, worst 109.5 s, P family 40/26%, 6-net bill -0.585% -> 1.4616e-9; VQCpbm cbc6cd11: 45,048/43,393, median .3399, worst 108.8 s, P family 34/23%, -0.531% -> 1.4624e-9.

## Programs

- [cand_frontier_VA_v1.py](cand_frontier_VA_v1.py)
- [cand_frontier_VAB_v1.py](cand_frontier_VAB_v1.py)
- [cand_frontier_VC_v1.py](cand_frontier_VC_v1.py)
- [cand_repair_VQpbm_v1.py](cand_repair_VQpbm_v1.py)
- [cand_repair_VQCpbm_v1.py](cand_repair_VQCpbm_v1.py)
- [../2026-10-16-oct-16-fire-rules-slot-9-only-clean-stack/eval_bar_v1.py](../2026-10-16-oct-16-fire-rules-slot-9-only-clean-stack/eval_bar_v1.py)
- [compare6_v1.py](compare6_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4358](../../TEAM_LOG_2026-10-01_to_10-22.md#L4358)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4360](../../TEAM_LOG_2026-10-01_to_10-22.md#L4360)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4371](../../TEAM_LOG_2026-10-01_to_10-22.md#L4371)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4376](../../TEAM_LOG_2026-10-01_to_10-22.md#L4376)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4378](../../TEAM_LOG_2026-10-01_to_10-22.md#L4378)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4399](../../TEAM_LOG_2026-10-01_to_10-22.md#L4399)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4430](../../TEAM_LOG_2026-10-01_to_10-22.md#L4430)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4433](../../TEAM_LOG_2026-10-01_to_10-22.md#L4433)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4445](../../TEAM_LOG_2026-10-01_to_10-22.md#L4445)

<details>
<summary>All result files and records (4)</summary>

- [gates_VC.txt](gates_VC.txt) · result
- [gates_VAB.txt](gates_VAB.txt) · result
- [gates_repvqpbm_20261016_opus_v1.txt](gates_repvqpbm_20261016_opus_v1.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Submission #336570: join ordering and three cutoff changes; a public network failed at .430 s residual](../../../submissions/336570/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-score-versus-runtime-risk-comparison-projected-1-0-score) · [Copy provenance](../../../PROVENANCE.json)

