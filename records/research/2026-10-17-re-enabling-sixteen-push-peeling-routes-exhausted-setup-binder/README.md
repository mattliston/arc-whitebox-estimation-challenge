# Re-enabling sixteen push-peeling routes exhausted the setup binder and raised a missing-read-key error

2026-10-17T20:45:00Z · incident · failed · measured

Rebinding sixteen non-slow push-peeling families consumed the setup budget before a required read route was bound. Prediction raised a missing-key error, so the arithmetic saving was blocked by both setup and wall limits.

Recorded measurement:

    KeyError (928,8128,896,0) in _PEEL_READ_IO; earlier full-binding starts 4.87–4.99 s against the 5 s cap.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4920](../../TEAM_LOG_2026-10-01_to_10-22.md#L4920)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4921](../../TEAM_LOG_2026-10-01_to_10-22.md#L4921)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-re-enabling-sixteen-push-peeling-routes-exhausted-setup-binder) · [Copy provenance](../../../PROVENANCE.json)

