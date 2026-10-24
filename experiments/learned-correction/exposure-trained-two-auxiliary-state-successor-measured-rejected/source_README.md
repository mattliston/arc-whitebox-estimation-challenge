# Exposure-trained multistate residual test

This directory contains exactly one preregistered successor to the scalar
exposure-trained recurrence in `public_exposure_trained_residual_state_v1`.
It retains the explicit physical primary transport `Phi * (e @ W)` and adds
two per-unit learned auxiliary states.  Both auxiliary coordinates are
independently transported with the legal `Phi * W` operator, then coupled by
bounded local gates and a near-identity 2x2 mixing matrix.  The complete
three-channel recurrence remains inside the phase-2 compute headroom without
changing the Q24 host.

The single fixed run fits public `observer_fit` ranks `0:159` and computes
metrics once on the already-development-open ranks `160:199` after epoch 20.
It cannot read observer-selection targets or observer-fit targets at rank 200
or above.  Data-derived old-host tails are replaced by the corrected
near-floor-min host.  The process aborts if its observed RSS exceeds 8 GiB.

Run from this directory with the explicit two-thread caps:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
/Users/matt/anaconda3/bin/python3 -B run_experiment.py
```

The fixed primary gate is autonomous final MSE below `3.123872816e-8`; the
stretch gate is below `1e-8`.  `RESULT.json` records the same-checkpoint
teacher-forced result and a complete analytic deployment-cost upper bound.

## Result

The fixed epoch-20 autonomous rollout was finite and reached
`3.4518303896e-8`.  That is only a 0.5514% MSE improvement over the scalar
parent (`3.4709697956e-8`), so it fails the preregistered
`3.123872816e-8` primary gate and the `<1e-8` stretch gate.  The identical
checkpoint reached `1.1469792562e-8` with the true preceding primary
residual, leaving a 3.0095x autonomous/teacher gap.  Two independently
transported tangent-memory channels therefore do not materially solve the
autonomous state bottleneck under this frozen exposure schedule.

The validation host is `9.1654788988e-8`.  The analytic deployment upper
bound is `2,127,226,767,293` host-plus-correction FLOPs, leaving
`72,773,232,707` FLOPs below the `2.2e12` ceiling.  Peak observed RSS was
`1,668,268,032` bytes.  The result used no observer-selection targets and no
observer-fit targets at rank 200 or above.
