# Hosted wall time correlates with requests and tensor writes more strongly than FLOPs

2026-10-21T15:02:12Z · finding · informational · measured

The runtime decomposition identified kernel work and evaluator overhead as the main wall-time components. Request and tensor-write counts were better local wall proxies than FLOPs, motivating uniform reductions in read traffic.

Recorded measurement:

    - Hosted wall anatomy: kernel 91% of wall; VM: corr(wall,requests)=0.71, corr(wall,puts)=0.74, corr(wall,flops)=0.12; ~44k requests/~48k puts/~300 GB pushed per network. Telemetry: lead/runs/fable_oct21_final_v1/telemetry_3375{06,34}.json (fetch_telemetry_v1.py SID OUT).
    - 2026-10-21T14:57:11Z [fable-lead] HOSTED PER-NETWORK: #337534 vs #337506 kernel ratio mean 1.011 median 1.005 on 43 nets; idx3/idx4 +11.5%/+9.7% in-call plus ~10 s evaluator roundtrip overhead (random burst). #337506 public walls: median 102.6 s, max 114.9, 41/50 above 99 s -> a burst fails ~half of fires. Safe target: slowest public net <= 95 s (-17%), residual max <= 0.34 s. Telemetry files lead/runs/fable_oct21_final_v1/telemetry_3375{06,34}.json; fetcher fetch_telemetry_v1.py. Matt: engineer margin + gain into the next file; no fire without approval.

## Programs

- [wall_anatomy_v1.py](wall_anatomy_v1.py)
- [op_spread_v1.py](op_spread_v1.py)
- [fetch_telemetry_v1.py](fetch_telemetry_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7435](../../TEAM_LOG_2026-10-01_to_10-22.md#L7435)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7428](../../TEAM_LOG_2026-10-01_to_10-22.md#L7428)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-hosted-wall-time-correlates-requests-tensor-writes-more-strongly) · [Copy provenance](../../../PROVENANCE.json)

