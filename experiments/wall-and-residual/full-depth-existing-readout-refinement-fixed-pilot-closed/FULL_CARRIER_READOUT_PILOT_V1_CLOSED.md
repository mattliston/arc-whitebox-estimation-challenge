# Full-depth existing-readout refinement: fixed pilot closed

2026-09-05, after13:36PDT. Phase2 only. No candidate build or submission.

The five-epoch,910-coefficient pilot completed in1121.94seconds on one
nice10 CPU worker with3threads. Maximum process RSS10.61GB; swap stayed
5966.50MiB. It used the newly certified complete CP/star/trace derivative,
dynamic selections, and the frozen scalar-final bank head. The independent
NumPy evaluation agrees with the torch final MSE to at most0.0059%.

Actual NumPy bank-path holdout results, all versus existing documented1e9
means and the same net's original bank-path mean:

| New-fit holdout | Parent raw MSE | Final pilot raw MSE | Ratio |
| --- | ---: | ---: | ---: |
|35-series rank37|2.12811809538e-8|2.15163014792e-8|1.01104828|
|rank39|1.78616289101e-8|1.79584053087e-8|1.00541812|
|rank42|2.32244332166e-8|2.33509181514e-8|1.00544620|
|rank48|2.24393813739e-8|2.21423815482e-8|0.98676435|

Only1/4 improves and the mean does not improve. This fails the declared
15% mean gain plus improvement on every holdout. No radius, epoch, model
capacity or expanded-dataset sweep for this fixed-readout refinement.
The20% relative trust radius is active only at layer1; that does not justify
a radius sweep. No intermediate checkpoint is selected after the fact.

This is a negative optimization result with a faithful derivative, not the
earlier frozen-K3 transport failure. The full-carrier implementation remains
available for genuinely different state actions. It does not establish a
universal ceiling for every learned moment model. Both panels, measured
bill and deployment gates were intentionally not run for this failed pilot.

Payload SHA256:708d92593f299d708c6be53ae1220ff2a459fe9e699ca536a8d9936c2dfcb43e.
Receipts:full_carrier_readout_pilot_v1.json and
full_carrier_readout_actual_r37/r39/r42/r48_v1.json. All five epoch artifacts
are preserved. Usable official bank329926 remains4.98223146018e-9 adjusted.
