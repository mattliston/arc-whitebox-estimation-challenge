# Error after transport-gate correction: two thirds originates at layers 12-15; board transfer about 0.8x

2026-10-12T23:10Z · finding · informational · measured

radlab's per-layer budget with the isotropic carry (0.86 at L12-15): L15 20%, L14 19%, L13 16%, L12 12%; the final error is not mean-aligned (3.9% along the true mean direction); board calibration over candidate generation 50 (S50) + candidate generation 51 (S51) + skew- and kurtosis-corrected transport gate (gk) gives board / PyTorch 0.70 (se 0.29) vs raw.

Recorded measurement:

    born per layer (c_rand) L15 20, L14 19, L13 16, L12 12, L8-11 26%; gk's gain 91% born at L12-15; twin-vs-twin board se ~0.10%; planning: board = ~0.8 x raw torch pooled

## Programs

- [errbudget_v1.py](errbudget_v1.py)
- [errbudget_v2.py](errbudget_v2.py)
- [boardcal_v1.py](boardcal_v1.py)
- [../2026-10-12-refit-parity-correlation-0-893-size-1-029-research/candidate-f314d8ea.py](../2026-10-12-refit-parity-correlation-0-893-size-1-029-research/candidate-f314d8ea.py)
- [../../../submissions/335633/candidate-fb4456a7.py](../../../submissions/335633/candidate-fb4456a7.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1703](../../TEAM_LOG_2026-10-01_to_10-22.md#L1703)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1706](../../TEAM_LOG_2026-10-01_to_10-22.md#L1706)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1712](../../TEAM_LOG_2026-10-01_to_10-22.md#L1712)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1786](../../TEAM_LOG_2026-10-01_to_10-22.md#L1786)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1794](../../TEAM_LOG_2026-10-01_to_10-22.md#L1794)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1862](../../TEAM_LOG_2026-10-01_to_10-22.md#L1862)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1901](../../TEAM_LOG_2026-10-01_to_10-22.md#L1901)

<details>
<summary>All result files and records (4)</summary>

- [errbudget_k7_v2.txt](errbudget_k7_v2.txt) · result
- [gate_rdz_xfer_B_all.json](gate_rdz_xfer_B_all.json) · result
- [gate_rdz_xfer_K_all.json](gate_rdz_xfer_K_all.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Final-error attribution: 78% carried mean error; a layer-15-only change has about a 22% ceiling](../2026-10-10-final-error-attribution-78-carried-mean-error-layer-15/README.md)
- enables: [Final-layer hub quartic ages 3-4 at strength 0.5 adopted: -0.247% mean-log MSE on 24 networks](../2026-10-13-final-layer-hub-quartic-ages-3-4-strength-0/README.md)
- ← enables: [Skew- and kurtosis-corrected transport gate adopted: -0.348% pooled MSE on 24 fresh networks](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-error-after-transport-gate-correction-two-thirds-originates-layers) · [Copy provenance](../../../PROVENANCE.json)

