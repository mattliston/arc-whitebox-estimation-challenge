# Forty-five cheap readout features across layers 10-15: best leave-one-network-out gain -0.022%

2026-10-20T09:50:40Z · experiment · rejected · measured

On 129 networks outside the 32-network judge set, linear bypasses using cheap per-unit features did not predict the final residual. The best leave-one-network-out gain was -0.022%.

Recorded measurement:

    [fable-lead] 2026-10-20T09:50:40Z HEAD-INPUT SCREEN (native, 129 store nets 301-438 outside the judge 32; lead/runs/fable_oct19_kc_v1/native_state_probe_v1.py -> stateprobe/sp_<rank>.npz (per layer: the head's 30 inputs, 64-dim state in/out, delta, pair-map aux, K21 / C / dcov row energies, all-layer y + truth) + head_input_screen_v1.py): ~45 cheap per-unit candidates x layers 10-15 as a linear bypass delta_l += c sigma_l f, fitted on the FINAL residual through the mean-chain Jacobian with leave-one-net-out. Best LOO gain -0.022% (dvhat x phi/2s at L12), everything else |LOO| <= 0.01%, including genuinely new features (diag-transport k3 estimate: newness 0.87 vs the head's span). The cheap new-input route is CLOSED natively (matches the 10-10 torch screen: 'the cheap five -0.02%'). Scaling the late heads' own deltas (c ~ -0.1..-0.2) is worth < 0.01%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/native_state_probe_v1.py](code/native_state_probe_v1.py)
- [code/head_input_screen_v1.py](code/head_input_screen_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6165](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6165)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-forty-five-cheap-readout-features-across-layers-10-15) · [Copy provenance](../../../PROVENANCE.json)

