# Native-route acceptance requires correlation >= 0.70, size within 1/3..3 and delivery >= 0.5 on two networks

2026-10-12T22:50Z · lesson · informational · measured

A per-layer dial reaches refit-after-drop (RAD) only through a certified native route: the native prediction change against the PyTorch change on two networks must correlate >= 0.70, have size within 1/3..3x above a 2e-6 floor, and deliver D >= 0.5; birth scale coefficient (bscale) at L0 was vetoed (D 0.24), x2a1 at L7 accepted (D 0.82), k22 at L14 D 1.00.

Recorded measurement:

    k21n 7-10 corr 0.84-0.85 size 1.1-1.2; rsy 11-14 corr 0.98 size 1.01; iso 11-14 corr 0.83-0.89; bscale_0 D 0.242 / 0.279 vetoed; x2a1_7 D 0.820 / 0.830; pe3/pe4/k21k/ds D 0.995-1.000

## Programs

- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v2.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v2.py)
- [parity_verdict_v5.py](parity_verdict_v5.py)
- [../2026-10-12-quoted-verdict-text-triggers-uncertified-route-automation-patterns-must/routes_update_v4.py](../2026-10-12-quoted-verdict-text-triggers-uncertified-route-automation-patterns-must/routes_update_v4.py)
- [../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_k22_v1.py](../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_k22_v1.py)
- [../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_x2a_v1.py](../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_x2a_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1397](../../TEAM_LOG_2026-10-01_to_10-22.md#L1397)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1399](../../TEAM_LOG_2026-10-01_to_10-22.md#L1399)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1403](../../TEAM_LOG_2026-10-01_to_10-22.md#L1403)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1420](../../TEAM_LOG_2026-10-01_to_10-22.md#L1420)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1442](../../TEAM_LOG_2026-10-01_to_10-22.md#L1442)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1726](../../TEAM_LOG_2026-10-01_to_10-22.md#L1726)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1733](../../TEAM_LOG_2026-10-01_to_10-22.md#L1733)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1748](../../TEAM_LOG_2026-10-01_to_10-22.md#L1748)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1751](../../TEAM_LOG_2026-10-01_to_10-22.md#L1751)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1756](../../TEAM_LOG_2026-10-01_to_10-22.md#L1756)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1759](../../TEAM_LOG_2026-10-01_to_10-22.md#L1759)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1761](../../TEAM_LOG_2026-10-01_to_10-22.md#L1761)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1765](../../TEAM_LOG_2026-10-01_to_10-22.md#L1765)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1917](../../TEAM_LOG_2026-10-01_to_10-22.md#L1917)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1921](../../TEAM_LOG_2026-10-01_to_10-22.md#L1921)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1931](../../TEAM_LOG_2026-10-01_to_10-22.md#L1931)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1934](../../TEAM_LOG_2026-10-01_to_10-22.md#L1934)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1940](../../TEAM_LOG_2026-10-01_to_10-22.md#L1940)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1724](../../TEAM_LOG_2026-10-01_to_10-22.md#L1724)

<details>
<summary>All result files and records (2)</summary>

- [ROUTES.json](../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/ROUTES.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Native routes add pair-map skew, kurtosis, kink-plane and per-layer correction dials](../../../instruments/native-routes-add-pair-map-skew-kurtosis-kink-plane/README.md)
- shipped in: [Submission #335618: learned-array, head-row and per-layer strength refit in the lower-request form](../../../submissions/335618/README.md)
- ← enables: [Radial birth-weight route rejected: native change is 5.4-6.0x research change with correlation about 0.1](../2026-10-12-radial-birth-weight-route-rejected-native-change-5-4/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-native-route-acceptance-requires-correlation-0-70-size-within) · [Copy provenance](../../../PROVENANCE.json)

