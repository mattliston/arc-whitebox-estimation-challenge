# Joint native readout refit at layers 12-15: -0.66% in sample, nonnegative leave-one-network-out error

2026-10-20T09:51:14Z · experiment · rejected · measured

A joint 260-parameter ridge readout fit through the mean-chain Jacobian failed leave-one-network-out validation at every ridge value. The terminal-only fit also worsened held-out predictions.

Recorded measurement:

    [fable-lead] 2026-10-20T09:51:14Z JOINT NATIVE READOUT REFIT DEAD (lead/runs/fable_oct19_kc_v1/joint_readout_refit_v1.py on the 129-net state probe): refitting the heads' readouts (head_l, head_b_l; 65 params x layers 12-15 = 260) jointly on the final residual through the mean-chain Jacobian: in-sample -0.66% but leave-one-net-out +0.54% at ridge 1e-4, +0.12% at 1e-1, +0.00% at 10 (LOO >= 0 at EVERY ridge); layer 15 alone in -0.07% / LOO +0.04%. The native readouts are at their population optimum, as the 10-16 single-layer refits found. With the head-input screen (<= 0.02%) this closes every refit-free and linear-refit route on the native card: the remaining gap needs new physics plus a nonlinear joint refit (torch engine), which cannot be built, ported and verified before the deadline.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/joint_readout_refit_v1.py](code/joint_readout_refit_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6167](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6167)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-joint-native-readout-refit-layers-12-15-0-66) · [Copy provenance](../../../PROVENANCE.json)

