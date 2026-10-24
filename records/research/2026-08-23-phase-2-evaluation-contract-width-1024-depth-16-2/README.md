# Phase 2 evaluation contract: width 1024, depth 16, 2^41 metered FLOPs and 120 s wall

2026-08-23T22:30Z · decision · adopted · derived

The official Phase 2 launch announcement (AIcrowd topic 18197) was recorded as the binding authority: width 1024 and depth 16, a per-network budget of 2^41 = 2,199,023,255,552 metered FLOPs, a 120 s prediction wall, 8 GB solution memory, a 0.4 s residual wall allowed only for control flow, a 5 s setup limit, dataset revision v2-phase2 with flopscope 0.12.0 and whestbench 0.16.0, 10 submissions per team per day, and all submission-time numerical work through the flopscope client (numpy, SciPy, compiled kernels, threads, subprocesses and foreign-function calls prohibited). The updated rules were re-accepted; the active research directive stayed on a pure closure before any hybrid.

Recorded measurement:

    score = final_layer_raw_mse x max(0.1, flops_used / 2^41); forward FLOPs per sample 33,554,432; naive forward-sample upper bound 65,536; ground-truth samples per MLP 1,000,000,000; mini 100 and full 1000 disjoint MLPs; submissions close 2026-10-17 23:59 UTC

## Evidence

- [OFFICIAL_PHASE2_LAUNCH_UPDATE.md#L6](OFFICIAL_PHASE2_LAUNCH_UPDATE.md#L6)
- [OFFICIAL_PHASE2_LAUNCH_UPDATE.md#L25](OFFICIAL_PHASE2_LAUNCH_UPDATE.md#L25)
- [OFFICIAL_SOURCE_AUDIT.json#L35](../2026-08-23-order-2-hermite-chain-versus-matched-cost-monte-carlo/OFFICIAL_SOURCE_AUDIT.json#L35)
- [source_README.md#L12](../2026-08-23-width-1024-scaling-screen-fixed-rank-carriers-fit-dense/source_README.md#L12)

<details>
<summary>All result files and records (4)</summary>

- [OFFICIAL_SOURCE_AUDIT.json](../2026-08-23-order-2-hermite-chain-versus-matched-cost-monte-carlo/OFFICIAL_SOURCE_AUDIT.json) · config
- [OFFICIAL_PHASE2_LAUNCH_UPDATE.md](OFFICIAL_PHASE2_LAUNCH_UPDATE.md) · record
- [source_README.md](../2026-08-23-width-1024-scaling-screen-fixed-rank-carriers-fit-dense/source_README.md) · record
- [requirements-phase2.txt](requirements-phase2.txt) · config

</details>

## Related work

- ← enables: [Phase 2 migration reused cumulant identities and carrier formulas while discarding old fitted parameters](../2026-08-23-phase-2-migration-reused-cumulant-identities-carrier-formulas-while/README.md)

[Research graph](../../../graph/index.html#node=2026-08-23-phase-2-evaluation-contract-width-1024-depth-16-2) · [Copy provenance](../../../PROVENANCE.json)

