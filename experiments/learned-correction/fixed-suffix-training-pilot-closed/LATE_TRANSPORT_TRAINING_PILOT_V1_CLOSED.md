# Fixed suffix training pilot: closed

September 5, 2026. Five predeclared epochs completed in 51.97 seconds on
the sole MPS job, with 20 train / 4 held-out public networks. The exact
prefix through layer 8 remains byte-identical. Shadow adjoints and coupled
late-layer finite-difference fidelity had passed before this fit.

Actual full dynamic carrier, stored 1e9 mean targets, final MSE ratios to
the frozen full-final-feature parent:

| Held-out rank | Parent raw | New raw | Ratio |
| --- | --- | --- | --- |
| 37 | 2.15350826147e-8 | 2.15428219830e-8 | 1.00035938 |
| 39 | 1.77858665601e-8 | 1.77753040756e-8 | 0.99940613 |
| 42 | 2.31795683708e-8 | 2.32253199215e-8 | 1.00197379 |
| 48 | 2.22270149547e-8 | 2.18234341703e-8 | 0.98184278 |

Fails the >=8% reduction on every held-out net gate. No 200-net run,
epoch/LR sweep, scalar-final refit, port, panel or submission. The shadow
predicted these small changes closely; the failure is not a claimed
large shadow gain that disappeared on deployment. This closes the fixed
warm suffix fit, not all end-to-end algorithms.

Evidence: late_transport_training_pilot_v1.json,
late_transport_actual_verdict_v1.json and payload_late_transport_pilot_v1.npz.
Official bank remains 329926, adjusted 4.98223146018e-9; goal below 1e-9
and the 4e-9/3e-9/2e-9 milestone banks remain unfinished.
