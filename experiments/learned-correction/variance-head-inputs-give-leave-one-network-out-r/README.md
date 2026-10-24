# Variance head inputs give leave-one-network-out R^2 <= +0.01 at every layer and ridge

2026-10-20T10:06:22Z · experiment · rejected · measured

A variance head using pre-activation moments, the 64-dimensional learned state and prior head inputs could not predict pre-variance error. The pooled layers 10-14 model had mean R^2 -0.004.

Recorded measurement:

    [fable-lead] 2026-10-20T10:06:22Z VARIANCE HEAD DEAD BEFORE BUILD (lead/runs/fable_oct19_kc_v1/varhead_screen_v1.py on stateprobe_var/sp_{35,37,39,100}.npz + mcall pre-covariance truths, L9-14): the per-unit PRE-variance error dv (the -18/-26% non-co-adapted oracle channel; truth MC noise only 4-14% of dv^2) is unpredictable from everything the learned correction sees at the start of the layer: ridge on [1, pre mu/v/k3/k4, the 64-dim GNN state, the previous layer's 30 head inputs], leave-one-net-out R^2 <= +0.01 at every layer and every ridge (pooled L10-14 model: mean -0.004). A learned variance correction has no input to work from. With the per-unit feature regression (R^2 ~ 0), the pair-geometry K21 model (0%), the missing-term tests and the chain law, the variance channel is closed from every side.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/varhead_screen_v1.py](code/varhead_screen_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6173](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6173)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-variance-head-inputs-give-leave-one-network-out-r) · [Copy provenance](../../../PROVENANCE.json)

