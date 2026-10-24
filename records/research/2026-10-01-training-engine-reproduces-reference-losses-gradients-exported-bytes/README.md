# Training engine reproduces reference losses, gradients and exported bytes

2026-10-01T18:52Z · tool · informational · measured

A new training engine for the learned per-layer correction was certified against the reference trainer: losses of four networks with the untrained parameters bit-identical to the reference's logged rows, the gradient of rank 49 bit-identical in all 23 arrays and 10 dials, and the exporter reproduces the bank source byte for byte. The baseline on the 40 validation networks was recorded for all later refits.

Recorded measurement:

    four-network summed gradient norms 7.338744163513184 / 4.599334701658686 (equal to the reference); exporter sha a3efb3a1 byte for byte; baseline pooled final-layer MSE on 40 validation nets 1.695523e-8 (VAL-A 1.738865e-8, VAL-B 1.652182e-8), all-layer 6.654e-9

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5](../../TEAM_LOG_2026-10-01_to_10-22.md#L5)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7](../../TEAM_LOG_2026-10-01_to_10-22.md#L7)

<details>
<summary>All result files and records (4)</summary>

- [CERTIFY.json](CERTIFY.json) · result
- [baseline_valid40.json](baseline_valid40.json) · result
- [RESULT.md](RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes) · [Copy provenance](../../../PROVENANCE.json)

