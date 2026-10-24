# Native parity script falsely failed because it compared float64 output with float32 bytes

2026-10-02T19:58Z · incident · informational · measured

The fired file's independent parity check initially reported failure because native float64 output was compared with boundary float32 bytes without widening. Using the same dtype comparison as the worker-stage check restored byte and bill equality on networks 35 and 100; the wrong result was not retained.

Recorded measurement:

    19:57Z false failure; corrected parity bytes identical and bills equal on35/100; step1-vs-step2 negative control fails

## Programs

- [native_parity.py](native_parity.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [stage_b_native40.py](stage_b_native40.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L457](../../TEAM_LOG_2026-10-01_to_10-22.md#L457)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-native-parity-script-falsely-failed-because-it-compared-float64) · [Copy provenance](../../../PROVENANCE.json)

