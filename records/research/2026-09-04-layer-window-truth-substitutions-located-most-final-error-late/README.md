# Layer-window truth substitutions located most final error in late layers

2026-09-04 · finding · rejected · measured

LATE-WINDOW CARRY DEAD (no-drop residual carry, relaxed arena, RETRO_RESID_LAYERS 10-15 / 12-15): r35 host 8.68e-8 / 8.62e-8 (= deployed 8.62e-8), r100 7.80e-8 (deployed 7.70e-8); with the operator on 5.4e-8 / 4.1e-8 (in-span). The carry's value is early/mid (T-51 window 1-10); it does not reach the late-born per-unit errors that T-98 localizes. STATUS 18:40: the closure family's measured frontier on this machine is ~1.5-1.8e-8 adjusted; the 4e-9 class needs a per-unit (v,k3,k4) supply at the last 3-7 layers with alignment >= 0.9, which no instrument here produces (carry no, feed small, Monte Carlo (MC) too noisy at 0.1%, the full tensor 0.5 util per layer).

Recorded measurement:

    - T-98 THE FINAL ERROR IS BORN LATE (r35, deployed card, F2J OFF, exact truth (v,k3,k4) states inside a layer window only, RETRO_FORCE_LAYERS; host 8.622e-8): windows 1-4 6.76e-8 (-22%), 1-8 5.55e-8 (-36%), 1-12 2.71e-8 (-69%), 13-15 4.39e-8 (-49%), 9-15 1.22e-8 (-86%), 5-15 ~0 (-100%), 1-15 ~0. Layers 1-4 contribute nothing; the last three layers' per-unit state errors are half the host error, the last seven 86%. (With the operator ON every full-truth window is worse, 2.86-4.7e-8: in-span/co-adaptation.) => any expensive per-unit supply needs to run only at the last 3-7 layers. Testing the no-drop residual carry on LATE windows with the operator kept.
    
    - T-98b LATE-WINDOW CARRY DEAD (no-drop residual carry, relaxed arena, RETRO_RESID_LAYERS 10-15 / 12-15): r35 host 8.68e-8 / 8.62e-8 (= deployed 8.62e-8), r100 7.80e-8 (deployed 7.70e-8); with the operator on 5.4e-8 / 4.1e-8 (in-span). The carry's value is early/mid (T-51 window 1-10); it does not reach the late-born per-unit errors that T-98 localizes. STATUS 18:40: the closure family's measured frontier on this machine is ~1.5-1.8e-8 adjusted; the 4e-9 class needs a per-unit (v,k3,k4) supply at the last 3-7 layers with alignment >= 0.9, which no instrument here produces (carry no, feed small, MC too noisy at 0.1%, the full tensor 0.5 util per layer).

## Evidence

- [LANE_g3_tracker.md#L774](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L774)
- [LANE_g3_tracker.md#L775](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L775)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-layer-window-truth-substitutions-located-most-final-error-late) · [Copy provenance](../../../PROVENANCE.json)

