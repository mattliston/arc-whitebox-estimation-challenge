# Rank-384 compact-layout stack fails request and slow-setup gates; exact plan repairs cost too much

2026-10-16T01:21:43Z · incident · rejected · measured

Compact layout left the rank 384 request count above the limit and added 95 setup requests. Restoring request prices or capacity consumed more bill than the pooled accuracy advantage.

Recorded measurement:

    risk22.44%;worst106.2s;setup4.461s,slow4.611s;repairs +382/+1.97%,+789/+2.19%,+341/+1.71%,+521/+2.10%;neutralizing>=~.4%bill

## Programs

- [../../../experiments/join-sketch/layer-10-join-rank-384-24-network-noise-free/code/census_v1.py](../../../experiments/join-sketch/layer-10-join-rank-384-24-network-noise-free/code/census_v1.py)
- [../../../experiments/join-sketch/layer-10-join-rank-384-24-network-noise-free/code/grade_sci24_v1.py](../../../experiments/join-sketch/layer-10-join-rank-384-24-network-noise-free/code/grade_sci24_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3631](../../TEAM_LOG_2026-10-01_to_10-22.md#L3631)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3633](../../TEAM_LOG_2026-10-01_to_10-22.md#L3633)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [F384_DECISION_v1.json](../../../experiments/join-sketch/layer-10-join-rank-384-24-network-noise-free/F384_DECISION_v1.json) · result
- [F384_STACK_DECISION_v1.json](../../../experiments/join-sketch/layer-10-join-rank-384-24-network-noise-free/F384_STACK_DECISION_v1.json) · result
- [SCI24_DECISION_v1.json](../../../experiments/join-sketch/layer-10-join-rank-384-24-network-noise-free/SCI24_DECISION_v1.json) · result
- [JUDGE_OFFICIAL120_f384_v1.txt](../../../experiments/join-sketch/layer-10-join-rank-384-24-network-noise-free/JUDGE_OFFICIAL120_f384_v1.txt) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-rank-384-compact-layout-stack-fails-request-slow-setup) · [Copy provenance](../../../PROVENANCE.json)

