# Joint mean/covariance fit fails; accelerator-native discrepancy is only 0.217-0.338% of error RMS

2026-09-20 · finding · informational · measured

The four stored MPS training-endpoint predictions and the native submission predictions of the old joint P/Q (mean-head and covariance) fit differ by only 0.217% to 0.338% of the native error RMS, and triangle-inequality bounds put the training-endpoint MSE at least 2.0444, 2.4082, 1.2732 and 1.4116 times the P-only MSE on nets 37/39/42/48, so the saved endpoint had already failed on its own training backend. All 14 post-initial online joint batch losses exceeded P-only after a 24.03x jump on batch 2; no optimizer bug or capacity bound is established and the old fit stays closed.

Recorded measurement:

    backend RMS gap 0.217%-0.338% of native error RMS; training-endpoint/P-only MSE lower bounds 2.0444, 2.4082, 1.2732, 1.4116 (nets 37/39/42/48); 14/14 post-initial batches worse, 24.03x jump on batch 2

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L10](RESULT.md#L10)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · result
- [result.json](result.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-joint-mean-covariance-fit-fails-accelerator-native-discrepancy-only) · [Copy provenance](../../../PROVENANCE.json)

