# Three-coefficient output calibration (y, mean y, y^2/mean y) fitted on 16 networks: LOO -0.78% but fresh +0.88%, closed

2026-10-14T18:30:43Z · experiment · rejected · measured

A frozen per-network output calibration with three exact terms fitted by ordinary least squares on 16 training networks looked good under leave-one-network-out but lost on 16 fresh networks, alone and composed with the energy-PCA compression. A cached-residual audit then showed the spatial-mean oracle share of the error is below the Monte Carlo noise share, so no cheap output calibration can help.

Recorded measurement:

    LOO geometric MSE -0.783079%; native 136 implementation PASS (+51,208 FLOPs, +15 ops); fresh 16 adjusted +0.879698738%; AEP2+calibration composition +0.644433549%; residual audit on 8 cached networks: spatial-mean oracle error share 0.807% < MC mean-noise share 0.935%; LOO constant calibration +0.196% MSE, centered cubic +0.578%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_output_calibration_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fit_output_calibration_v1.py](code/fit_output_calibration_v1.py)
- [code/output_calibration_fragment_v1.py](code/output_calibration_fragment_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2963](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2963)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2966](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2966)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2968](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2968)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2970](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2970)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3159](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3159)

<details>
<summary>All result files and records (3)</summary>

- [FIT_AND_NETWORK_LOO_v1.json](FIT_AND_NETWORK_LOO_v1.json) · result
- [CALIBRATION_AND_FIXED_STACK_FRESH_v1.json](CALIBRATION_AND_FIXED_STACK_FRESH_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-three-coefficient-output-calibration-y-mean-y-y-2) · [Copy provenance](../../../PROVENANCE.json)

