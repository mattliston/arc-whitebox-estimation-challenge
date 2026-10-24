# Residual-corrector refit specified three collective-tracker features in addition to the 20-field tape

2026-08-31 · decision · informational · measured

P2R0F2Q BUILD PLAN (mechanical, ready to execute): the fitted transport-energy residual correction (F2J) corrector consumes the raw20 per-unit tape (LOCAL_FEATURES=20, loader public_full688_local_innovation_transport_v1/dataset.py) + 4x4 channel state; fit machinery torch, 688-net folds, prereg discipline. AUGMENT 20->23: per (layer, unit) add [K3S_l * w_i^2 (collective diag image), w_i (loading), delta_l * b_i (variance-direction load)] - all computed from the tracker + closure (m, C) states. TAPE ECONOMICS SOLVED: full host replays for 688 nets would be ~17 h, but the tracker needs only the m/C recursion (~7e10 flops/net = seconds on MPS) -> a light state-replay pass generates the 3 new columns for all 688 nets in ~2-4 h.

Recorded measurement:

    P2R0F2Q BUILD PLAN (mechanical, ready to execute): the F2J corrector consumes the raw20 per-unit tape (LOCAL_FEATURES=20, loader public_full688_local_innovation_transport_v1/dataset.py) + 4x4 channel state; fit machinery torch, 688-net folds, prereg discipline. AUGMENT 20->23: per (layer, unit) add [K3S_l * w_i^2 (collective diag image), w_i (loading), delta_l * b_i (variance-direction load)] - all computed from the tracker + closure (m, C) states. TAPE ECONOMICS SOLVED: full host replays for 688 nets would be ~17 h, but the tracker needs only the m/C recursion (~7e10 flops/net = seconds on MPS) -> a light state-replay pass generates the 3 new columns for all 688 nets in ~2-4 h. Then: refit (their machinery), package payload, card screen (harness standing, exact-reproduction verified), 8-net panel, fire under the meaningful-gain bar. This is the one route that survives all five co-adaptation lessons.

## Programs

- [../../../experiments/flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/code/public_full688_local_innovation_transport_v1/dataset.py](../../../experiments/flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/code/public_full688_local_innovation_transport_v1/dataset.py)

## Evidence

- [dataset.py#L1](../../../experiments/flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/code/public_full688_local_innovation_transport_v1/dataset.py#L1)

[Research graph](../../../graph/index.html#node=2026-08-31-residual-corrector-refit-specified-three-collective-tracker-features-addition) · [Copy provenance](../../../PROVENANCE.json)

