# Transported per-unit state corrector failed held-out correlation gates

2026-09-03 · experiment · rejected · measured

TRANSPORTED STATE-CORRECTOR GATE: DEAD (transported_dv_gate.py; derived host = chain + carry on the 8 panel nets, feat_derived_retro-*.npz; ridge, leave-one-net-out, held-out corr mean/min): dv local 0.17/0.14 (L10) .. 0.19/0.04 (L14); + the ORACLE previous-layer dv transported through W^2 (three gate weightings) 0.17-0.20 - the true previous-layer variance error does not predict the next layer's; the autonomous rollout equals local. dk3 local 0.35-0.42 at L10-14; dk4 local 0.55-0.64. Nothing near the 0.65-0.9 bars: the per-unit variance error is born fresh at each layer (T-80c's four-neuron content), not transported. 8-NET DERIVED-HOST PANEL (chain + carry, no feed, fitted transport-energy residual correction (F2J) off, util 0.82-0.84): 2.39 / 3.16 / 3.14 / 3.39 / 2.36 / 2.31 / 4.14 / 3.28e-8 (35/100/150/200/250/300/400/500), mean 3.02e-8: not a bank. STATE (17:00): no measured route to 4e-9 on this machine; the competitor's phase-2 method is unknown (his phase-1 cubature class measures 2.2x over Monte Carlo (MC) at this shape, T-93b).

Recorded measurement:

    - T-94 TRANSPORTED STATE-CORRECTOR GATE: DEAD (transported_dv_gate.py; derived host = chain + carry on the 8 panel nets, feat_derived_retro-*.npz; ridge, leave-one-net-out, held-out corr mean/min): dv local 0.17/0.14 (L10) .. 0.19/0.04 (L14); + the ORACLE previous-layer dv transported through W^2 (three gate weightings) 0.17-0.20 - the true previous-layer variance error does not predict the next layer's; the autonomous rollout equals local. dk3 local 0.35-0.42 at L10-14; dk4 local 0.55-0.64. Nothing near the 0.65-0.9 bars: the per-unit variance error is born fresh at each layer (T-80c's four-neuron content), not transported. 8-NET DERIVED-HOST PANEL (chain + carry, no feed, F2J off, util 0.82-0.84): 2.39 / 3.16 / 3.14 / 3.39 / 2.36 / 2.31 / 4.14 / 3.28e-8 (35/100/150/200/250/300/400/500), mean 3.02e-8: not a bank. STATE (17:00): no measured route to 4e-9 on this machine; the competitor's phase-2 method is unknown (his phase-1 cubature class measures 2.2x over MC at this shape, T-93b).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g29_bond_kernel_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/transported_dv_gate.py](code/transported_dv_gate.py)

## Evidence

- [LANE_g3_tracker.md#L767](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L767)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-03-transported-per-unit-state-corrector-failed-held-out-correlation) · [Copy provenance](../../../PROVENANCE.json)

