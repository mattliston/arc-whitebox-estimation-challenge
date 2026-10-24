# GPU experiments re-enabled with one run at a time and memory-pressure checks

2026-09-24T21:26Z · decision · informational · measured

The repository owner re-authorized GPU use, subject to serialized runs and memory checks.

Recorded measurement:

    - 2026-09-24 14:26 lead: MATT RE-ALLOWED MPS ("you can use mps again"). Rule: one MPS run at a time, float32, check swap. Queue: (1) physicist's dense full-cum3 carry-term sizing (A card transport vs B + distinct-index d_i d_j C_ij k3 g_k, vs 8M truth planes, r35), peak <= 10 GB; (2) covhunter's capacity follow-ups (narrow-control extra epoch, 4x width, wide2x on TRAIN640) ONLY if the CPU wide2x run (~15:45) gains > 0.3% on VALID16. The T640 CPU refits run to completion.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L408](../../TEAM_LOG_2026-09-22_to_09-26.md#L408)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-gpu-experiments-re-enabled-one-run-time-memory-pressure) · [Copy provenance](../../../PROVENANCE.json)

