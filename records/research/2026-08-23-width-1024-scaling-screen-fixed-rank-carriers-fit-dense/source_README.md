# Phase 2: width 1024, depth 16

This directory is the clean migration boundary between the frozen Phase 1
research record and the intended Phase 2 problem.

As of 2026-08-23 the versioned public assets are available, but AIcrowd has
not formally opened Phase 2 submissions: the evaluator is still undergoing a
security review.  Everything here is therefore a public-prelaunch research
program.  The official announcement can supersede limits recorded here.

## What changed

- MLP shape: width 256 / depth 32 -> width 1024 / depth 16.
- Per-MLP budget: 272,000,000,000 -> 2^41 = 2,199,023,255,552 FLOPs.
- Meter: flopscope 0.10.0 -> 0.12.0.
- Harness: whestbench 0.14.0 -> 0.16.0.
- Residual Python time: previously priced, now prohibited as a numerical
  compute channel and hard-capped at 0.4 seconds per prediction.
- Estimator memory: 8 GiB; setup: 5 seconds; prediction: 120 seconds.

The budget and a dense forward pass both grow by about eight times, so a
fixed-rank `O(depth * width^2 * rank)` carrier is almost budget-neutral.
Dense cubic work is not: it becomes roughly 3.8--4.0 times more expensive
relative to budget.  This makes compressed CP/TT propagation more attractive
while making the old dense Cholesky/PSD repair the first implementation
bottleneck.

## What carries over

The mathematical identities and failure mechanisms carry over:

- centered CP3/TT3 transport and K21 action identities;
- exact K21 overlays and the correlation-interior PSD construction;
- scalar Edgeworth readout from mean, variance, K3, and local K4;
- the need for a coherent joint state rather than an isolated improved
  third-order coordinate;
- the distinction between fixed-state imitation and autonomous closed-loop
  stability;
- target-blind prediction seals and split discipline.

## What does not carry over

No Phase 1 score, checkpoint, FLOP receipt, rank, cap, layer schedule, split,
or trajectory is a Phase 2 result.  In particular:

- layer 32 no longer exists;
- the aligned 32-channel observer was trained around the width-256/depth-32
  state distribution;
- rank 22 and the CAP3=1024 policy were selected under different geometry;
- all old metering predates flopscope 0.12.0;
- the old tensor host includes a dense cubic PSD certificate that screens
  over budget when naively migrated.

## Current hypothesis and order of work

The new geometry should favor cumulant propagation: wider layers are closer to
the Gaussian/low-cumulant regime and sixteen ReLU resets provide fewer chances
for closure error to compound.  A public participant measurement supports this
directly: a zero-fit second-order analytic chain moved from roughly parity with
FLOP-matched Monte Carlo at 256x32 to an analytic/MC MSE ratio of 0.12 at
1024x16.

The Phase 2 order is therefore:

1. pin the public authority and meter;
2. establish fresh diagonal-Gaussian and Hermite/Mehler baselines;
3. re-meter and simplify the CP3/TT3 host, especially the cubic PSD repair;
4. select caps/ranks from Phase 2 source gates rather than inherit rank 22;
5. re-run teacher substitutions to locate the new mean/variance/K3/K4 floor;
6. only then retrain an observer end to end on Phase 2 trajectories;
7. retain whitening/QMC as a measured comparator and possible hybrid lane.

See `OFFICIAL_SOURCE_AUDIT.json`, `MIGRATION_MATRIX.md`, and
`STATIC_SCALING_GATE.json` for the bound evidence.

