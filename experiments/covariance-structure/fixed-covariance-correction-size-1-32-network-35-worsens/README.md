# Fixed covariance correction of size 1/32 on network 35 worsens adjusted error: ratio 1.1486 versus gate 0.98

2026-09-20 · experiment · rejected · measured

The one prospectively selected finite endpoint of the covariance direction study, the unchanged +1/32 source generated before the response calibration, was scored on net 35 with its already saved output: raw MSE rose from 1.8706028490077942e-8 to 2.1480147674919686e-8 and the own-bill adjusted ratio of 1.148604184847961 fails the fixed <=0.98 gate, so no sign or amplitude search followed.

Recorded measurement:

    parent raw 1.8706028490077942e-8, bill 398080917697 F, adjusted 3.3862820545420866e-9; candidate raw 2.1480147674919686e-8, bill 398186088371 F, adjusted 3.889497738922592e-9; raw ratio 1.1483008104212604; adjusted ratio 1.148604184847961 vs <=0.98

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/covariance_fixed_endpoint_objective_v1/score35.py](code/covariance_fixed_endpoint_objective_v1/score35.py)
- [code/covariance_direction_native_v4/probes/plus.py](code/covariance_direction_native_v4/probes/plus.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L7](RESULT.md#L7)
- [RESULT.md#L8](RESULT.md#L8)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · result
- [result.json](result.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-fixed-covariance-correction-size-1-32-network-35-worsens) · [Copy provenance](../../../PROVENANCE.json)

