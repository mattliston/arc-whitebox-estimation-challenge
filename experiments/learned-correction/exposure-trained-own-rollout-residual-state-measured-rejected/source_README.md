# Exposure-trained residual-state test

This directory contains one bounded, preregistered test of whether the compact
6,688-parameter local-innovation recurrence improves when trained under its
own rollout distribution.  The fit split is observer-fit ranks `0:160`;
observer-fit ranks `160:200` are evaluated once after the fixed epoch.  The
complete `0:199` target prefix was materialized at startup after the schedule
was preregistered, so `160:199` is a development-validation panel, not a fresh
holdout.  No observer-selection target or observer-fit target at rank 200 or
above is read.

The schedule transitions from teacher exposure to fully autonomous exposure
and ends with eight fully autonomous epochs.  The exact `Phi * (e @ W)`
transport remains in the recurrence.  Old-host tails are identified solely by
the registered `5e-7` score threshold and are replaced by the corrected
near-floor-min host.

Run with the explicit two-thread caps:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
/Users/matt/anaconda3/bin/python3 -B run_experiment.py
```

`RESULT.json` and `MODEL.pt` are written once.  The run refuses to overwrite
them.  The primary gate is a finite autonomous rollout with validation final
MSE below `3.44e-8`; the `<1e-8` stretch result is recorded separately.

## Result

The autonomous rollout was finite and reached `3.4709698e-8`, a 9.14%
improvement over the established `3.82e-8` autonomous family, but it missed
the preregistered `3.44e-8` gate by 0.90% and did not reach `1e-8`.  The same
checkpoint with the true preceding residual supplied at each transition
reached `8.2079635e-9`.  Exposure training therefore nearly met the incremental
gate, while the remaining 4.23x autonomous/teacher gap still identifies
residual-state observability/transport as the limiting leaf.
