# Apparent terminal skew gain of -7.78% came from a contaminated baseline

2026-10-20T01:52:00Z · incident · informational · measured

A fresh-network direction sweep reported the terminal k3 scale 0.98 at raw -7.78% (6/6 networks) and a z-tilt at -5.84%, and store arms plus release forms were launched on it, before it was found that the sweep's candidate kept the full truth correction (strength 1.0, +116%) as its default, so every reading was relative to an exploded base. On the clean base the terminal k3 scale is at its optimum. Rule: always check the base U of an in-process evaluation against the known base (7100005: 1.3206e-8) before believing a delta.

Recorded measurement:

    contaminated: scale 0.98 -7.78% (6/6), 1.02 +8.55% (0/6), tilt -0.0007 -5.84%; base U 2.86e-8 vs true 1.32e-8; clean: 0.98 +0.5..+1.0%, 1.02 +0.3..+0.6%, 0.95 +3.3..+4.9%, tilt +0.8..+1.0%

## Programs

- [candidate_k3l15dir_sci_v1.py](candidate_k3l15dir_sci_v1.py)
- [../../../experiments/moment-propagation/truth-directed-cumulant-corrections-worsen-raw-error-68-105/code/candidate_k3l15dir2_sci_v1.py](../../../experiments/moment-propagation/truth-directed-cumulant-corrections-worsen-raw-error-68-105/code/candidate_k3l15dir2_sci_v1.py)
- [candidate_rel_v4b_s090.py](candidate_rel_v4b_s090.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6101](../../TEAM_LOG_2026-10-01_to_10-22.md#L6101)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6105](../../TEAM_LOG_2026-10-01_to_10-22.md#L6105)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6109](../../TEAM_LOG_2026-10-01_to_10-22.md#L6109)

<details>
<summary>All result files and records (3)</summary>

- [eval_k3l15dir_v1.log](eval_k3l15dir_v1.log) · log
- [eval_k3l15scale_v1.log](eval_k3l15scale_v1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-apparent-terminal-skew-gain-7-78-came-contaminated-baseline) · [Copy provenance](../../../PROVENANCE.json)

