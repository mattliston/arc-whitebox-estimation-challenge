# Read-only SSH monitor misclassified as heavy work after a completed 32-network judge

2026-10-21T04:38:25Z · incident · informational · measured

All predictions, guards and scoring finished successfully, but cleanup rejected a small read-only monitoring process. The same-identity queue was restored and the monitor stopped; numerical evidence and original guards were preserved.

Recorded measurement:

    Energy E2/E3 currentGram Mac32 completed04:31:24Z at+.065519076% adjusted/held+.031493298%,13/32lower,bill+.000103863%; no promotion, VM135738/scorer136736 finish normally. Source56c1cae6. Mac wrapper session71052 EXIT1 only in finally: deny_external_heavy falsely classified local read-only SSH truth monitor70377 as native. All32 actual predictions/12 guards/result complete and valid. Root inspected entire monitor (metadata only,11MBRSS), restored same-identity77740 at04:34:40Z, wrote energy output_dispatch/ROOT_CLEANUP_REPAIR_v1.json, preserved original STARTED dispatcher record and all original code/guards. Joint stopped only local monitor70377 at04:34:54Z; remote truth unaffected. Do not restart matching local monitors while Mac wrappers active. W3 fully CLOSED both hosts: Linux+.051329717%/held+.079260307%, Mac+.101230245%/held+.107566543%, bill0; closure40711c40, no sweep.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6593](../../TEAM_LOG_2026-10-01_to_10-22.md#L6593)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ROOT_CLEANUP_REPAIR_v1.json](../../../experiments/learned-correction/current-message-edge-context-refit-current-consumer-projector-metric/ROOT_CLEANUP_REPAIR_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-read-only-ssh-monitor-misclassified-heavy-work-after-completed) · [Copy provenance](../../../PROVENANCE.json)

