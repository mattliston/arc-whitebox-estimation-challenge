# Native carried-weight and radial readout hook passes its unit check at 0.003% FLOP cost

2026-10-10T02:26Z · tool · informational · measured

The native readout fragment records each layer state and applies learned carried-weight and radial updates only after that layer has consumed its state. A unit transcription matched to the last bit and a random-weight native smoke test completed; subsequent line searches determined accuracy separately.

Recorded measurement:

    +254 requests a network (18 a layer for both), +7.5 Mflop (0.003% of bill)

## Programs

- [code/fragment_ro_v1.py](code/fragment_ro_v1.py)
- [code/unit_ro.py](code/unit_ro.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L487](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L487)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-10-native-carried-weight-radial-readout-hook-passes-unit-check) · [Copy provenance](../../PROVENANCE.json)

