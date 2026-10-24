# Fresh-network Monte Carlo truth production: byte-identical synchronization reduction gives 1.027x throughput

2026-10-12T16:23Z · tool · informational · measured

Generated networks with Monte Carlo truths became the judging currency: a 64M-sample network costs about 9 minutes of GPU, 16M about 2.2 minutes; the analyst produced blocks continuously, a chain ordered them by round need, and mc_truth_v2 removed 7 of 8 host syncs byte-identically.

Recorded measurement:

    16M truth noise 4.56e-9 vs fresh final MSE ~1.5e-8; 8M networks give a smaller paired se per GPU hour for c <= 1; v2 bit-identical, 64.6 -> 62.9 s per 8M network; 2,124 truths by 10-13 23:31Z

## Programs

- [code/fresh/mc_truth.py](code/fresh/mc_truth.py)
- [code/fresh/mc_truth_v2.py](code/fresh/mc_truth_v2.py)
- [code/fresh/profile_mc_v1.py](code/fresh/profile_mc_v1.py)
- [code/fresh/next_block_v2.py](code/fresh/next_block_v2.py)
- [code/a27_fast_truth_20261011T0023Z/produce_v1.py](code/a27_fast_truth_20261011T0023Z/produce_v1.py)
- [code/fresh/loader.py](code/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L818](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L818)
- [TEAM_LOG_2026-10-01_to_10-22.md#L820](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L820)
- [TEAM_LOG_2026-10-01_to_10-22.md#L834](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L834)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1000](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1000)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1341](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1341)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1342](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1342)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1466](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1466)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1501](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1501)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1502](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1502)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1526](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1526)

<details>
<summary>All result files and records (3)</summary>

- [BITIDENT_mc_truth_v2.json](BITIDENT_mc_truth_v2.json) · result
- [plan.json](../variable-sample-truth-consumer-validates-sample-counts-sealed-network/plan.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Untouched generated networks become the primary judge because reused ranks overstate refit gains](../../records/research/2026-10-10-untouched-generated-networks-become-primary-judge-because-reused-ranks/README.md)

[Research graph](../../graph/index.html#node=2026-10-12-fresh-network-monte-carlo-truth-production-byte-identical-synchronization) · [Copy provenance](../../PROVENANCE.json)

