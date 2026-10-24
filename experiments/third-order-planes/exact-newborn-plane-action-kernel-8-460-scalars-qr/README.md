# Exact newborn-plane action kernel (8,460 scalars) on the QR read: pilot-8 net +0.20% vs QR, +0.16% vs fired file, closed

2026-10-15T14:26:07Z · experiment · rejected · measured

An exact newborn-plane (NP) hub-action kernel (cost 0.159% of bill in the tiny tariff) added on top of the QR read cost more bill than it returned in accuracy on the pilot.

Recorded measurement:

    vs QR: MSE +0.03477308578857967%, bill +0.16142411727938882%, net +0.19619720306796848%, ops +559; vs e406: MSE -0.04032560737880409%, bill +0.20241307587317006%, net +0.16208746849436595%, ops +570.875

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_np_action_rpc_reduction_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/kernel_np_hub_actions_HELD_v1.py](code/kernel_np_hub_actions_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3378](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3378)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3385](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3385)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3387](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3387)

<details>
<summary>All result files and records (2)</summary>

- [TINY_HUB_ACTIONS_AND_TARIFF_v1.json](TINY_HUB_ACTIONS_AND_TARIFF_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-exact-newborn-plane-action-kernel-8-460-scalars-qr) · [Copy provenance](../../../PROVENANCE.json)

