# Native covariance arithmetic causes the first large finite-response instability

2026-09-27 · finding · informational · measured

Passive traces and same-input replays localized the first response instability to covariance arithmetic. Ordinary float32 and float64 products largely removed the layer9 gap, while the star response was much smaller.

Recorded measurement:

    Native pre9 covariance secant gaps81.41/85.11%;ordinary FP32 replay0.7004/0.8129%,diagonal2.25/2.48%;FP64 linear to1.2e-11;native star4.73/5.38%.

## Programs

- [analyze.py](analyze.py)
- [covariance_replay.py](covariance_replay.py)
- [launch_trace.py](launch_trace.py)
- [localize.py](localize.py)
- [trace.py](trace.py)

## Evidence

- [RESULTS.md#L73](../../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L73)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-native-covariance-arithmetic-causes-first-large-finite-response-instability) · [Copy provenance](../../../PROVENANCE.json)

