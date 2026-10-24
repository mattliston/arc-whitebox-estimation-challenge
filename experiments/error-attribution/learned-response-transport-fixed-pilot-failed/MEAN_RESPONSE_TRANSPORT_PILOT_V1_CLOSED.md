# CLOSED — learned response transport, September6,2026

The response itself is measurably better at predicting gates: using two
independent2^20-sample streams each, late8..14 uniform cross-MSE learned/
Gaussian=.019153/.017730 and learned/host=.872946/.858712 on35/100.
The >10% host advantage has a narrow sampling margin (.68/.89 jackknife
standard errors at that threshold). No more gate sampling was performed.

Synthetic full-tensor repeated restoration passed1.26e-16. Both current
disabled paths were byte-identical to original P/Q trajectories and all
enabled P0 responses were captured before current mean corrections. Fixed
response changed only inherited CP legs8..14, original newborn geometry,
trace and declared repeated targets retained. Frozen final-mean shifts
were4.56e-6/4.07e-6 RMS on35/100, not accuracy measurements.

The predeclared matched20/4 two-stage pilot FAILED. Main P1 received one
40-epoch fit while auxiliary response P0 remained frozen with its own
honest recurrent state. One40-epoch final Q fit on actual on-policy rows.
Exactly24 teacher and24 final rows; no current-target feature leakage.
Actual final h/host/sigma/P1 state byte-identical to training row inputs.

Four fixed holdouts37/39/42/48, existing1e9 truths:

| Quantity | Pooled raw final MSE |
|---|---:|
| Original bank | 2.120165611358589e-8 |
| Immutable matched control | 2.235463499383767e-8 |
| Response transport | 2.247214819383274e-8 |

New/control=1.0052567711, new/bank=1.0599241905,0/4 wins. Thresholds
were<=.90 and<=.92 with>=3 wins. The control was reused with exact payload
hash checks and its archived MSE reproduced exactly, not retrained.

Thus accurate gates did not make this distinct-only inherited transport
more accurate end to end. No larger fit, layer/strength/epoch/seed/feature
sweep, deployment port, panel, or submission. Preserve all arrays/payloads.
Receipt: mean_response_transport_pilot_verdict_v1.json. Source transport
SHA138a59625b67ef0ddf7fa75768aae115dbf4835ea7702586a9eed60c78cea00a.
