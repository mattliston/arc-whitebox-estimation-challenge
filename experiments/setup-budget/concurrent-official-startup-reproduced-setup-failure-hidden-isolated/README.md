# Concurrent official startup reproduced a setup failure hidden by isolated qualification

2026-10-17T04:24:06Z · experiment · adopted · measured

The unchanged setup-failed source passed an isolated official startup but failed all five concurrent independent server and worker starts. Coalescing temporary-view cleanup removed hidden free requests and restored concurrent startup completion.

Recorded measurement:

    Isolated setup 4.183 s; concurrent SETUP_TIMEOUT 5.017–5.022 s; SDK imports 1.425 s; source load ~.197 s; cleanup removes 2961 hidden-free RPCs; repaired starts 4.857–4.876 s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_setup_repair_v18c.py](code/candidate_setup_repair_v18c.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4811](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4811)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4813](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4813)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [CONCURRENT_BASELINE_v1_SUMMARY.json](CONCURRENT_BASELINE_v1_SUMMARY.json) · result
- [CONCURRENT_BASELINE_v1_worker0.json](CONCURRENT_BASELINE_v1_worker0.json) · result
- [CONCURRENT_BASELINE_v1_worker1.json](CONCURRENT_BASELINE_v1_worker1.json) · result
- [CONCURRENT_BASELINE_v1_worker2.json](CONCURRENT_BASELINE_v1_worker2.json) · result
- [CONCURRENT_BASELINE_v1_worker3.json](CONCURRENT_BASELINE_v1_worker3.json) · result
- [CONCURRENT_BASELINE_v1_worker4.json](CONCURRENT_BASELINE_v1_worker4.json) · result

</details>

## Related work

- shipped in: [Submission #336640: fixed 224-key setup and batched aliases: setup failure before evaluation](../../../submissions/336640/README.md)
- shipped in: [Submission #336649: coalesced temporary-view cleanup: two cold public networks exceeded runtime caps](../../../submissions/336649/README.md)

[Research graph](../../../graph/index.html#node=2026-10-17-concurrent-official-startup-reproduced-setup-failure-hidden-isolated) · [Copy provenance](../../../PROVENANCE.json)

