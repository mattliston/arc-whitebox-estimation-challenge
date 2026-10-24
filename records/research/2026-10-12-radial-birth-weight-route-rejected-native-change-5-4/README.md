# Radial birth-weight route rejected: native change is 5.4-6.0x research change with correlation about 0.1

2026-10-12T04:15Z · incident · failed · measured

The candidate generation 50 (S50) round's kappa family had no native route; the same formula on the step's birth line moves native predictions 6x PyTorch's with correlation 0.1, so kappa is dropped from the port and from later arms (its shift was below the native rounding floor anyway).

Recorded measurement:

    corr +0.09 / +0.10, native change 6.0 / 5.4x torch; torch arm -0.014% mean MSE over 64 networks; S50's shift +0.0044 = 3e-7 of the prediction

## Programs

- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_kappa_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_kappa_v1.py)
- [../../../instruments/per-layer-carried-column-scale-hook-byte-identical-identity/code/opus_reexam_20261001_v1/engineer/a48_parity/parity_change.py](../../../instruments/per-layer-carried-column-scale-hook-byte-identical-identity/code/opus_reexam_20261001_v1/engineer/a48_parity/parity_change.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1305](../../TEAM_LOG_2026-10-01_to_10-22.md#L1305)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1306](../../TEAM_LOG_2026-10-01_to_10-22.md#L1306)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1312](../../TEAM_LOG_2026-10-01_to_10-22.md#L1312)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1317](../../TEAM_LOG_2026-10-01_to_10-22.md#L1317)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1322](../../TEAM_LOG_2026-10-01_to_10-22.md#L1322)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Native-route acceptance requires correlation >= 0.70, size within 1/3..3 and delivery >= 0.5 on two networks](../2026-10-12-native-route-acceptance-requires-correlation-0-70-size-within/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-radial-birth-weight-route-rejected-native-change-5-4) · [Copy provenance](../../../PROVENANCE.json)

