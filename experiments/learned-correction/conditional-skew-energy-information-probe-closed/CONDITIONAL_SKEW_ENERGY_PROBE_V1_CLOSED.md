# Conditional skew energy information probe: closed

September 5, 2026. Exactly the fixed top-eight physical covariance modes,
four rotation/sign-invariant regression statistics, 20train/4heldout and
40 epochs. All inputs came from previously leakage-checked teacher states.
No new current-estimator executions were needed for feature generation.

Held-out late-layer residual MSE, fixed epoch40:

- Frozen parent: 4.809338149947e-9.
- Same-capacity control: 4.819527776867e-9.
- Conditional-energy head: 4.820089327673e-9.
- Joint/control ratio 1.00011652; joint/parent 1.00223548; 0/4 nets improve.

The information gate fails. No larger rank, basis sweep, extra rows,
200-net fit, port, panel or submission for this fixed feature set.
Evidence: conditional_skew_energy_probe_rows_v1.json,
conditional_skew_energy_probe_verdict_v1.json, and saved model pair.
These are honest teacher residuals, not candidate adjusted scores.
