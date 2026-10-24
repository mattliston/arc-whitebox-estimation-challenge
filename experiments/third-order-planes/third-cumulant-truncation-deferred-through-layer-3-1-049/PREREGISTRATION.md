# CP1792 deferred-W3 truncation: target-free preregistration

## Question

Does the first CP truncation happen one layer too early?  The cap-1792 control
forms 2,048 terms after h2, selects 1,792 in h2 coordinates, and only then
applies W3.  This candidate carries all 2,048 terms through W3 and restores the
1,792 cap after the h3 gate and Gaussian-star birth.  It changes no later rank
cap, analytic covariance, mean recurrence, Hermite order, or K3 formula.

## Frozen intervention

- Width/depth: 1,024 by 16.
- Host: the target-blind cap-1792 estimator whose source SHA-256 is
  `035e3bd2fd08d3358f5424fc67f10281e5559e65fa9c6d6dce97791108b176dc`.
- h1 CP rank: 1,024 in both control and candidate.
- h2 CP rank: 1,792 in the control and 2,048 in the candidate.
- h3 and later CP cap: 1,792 in both.
- Selection law: the unchanged factor-energy ordering, but the candidate's
  first selection is made after W3, the h3 gate, and the h3 birth block.
- Targets, role labels, teacher states, fitted parameters, and prior scores:
  forbidden.

This gives an exact rank-2,048 CP K3 read at layer 3 under the shared analytic
host.  It is not an exact statement about the true network distribution.

## Resource hypothesis

The dominant extra action over cap-1792 is two float32 matrix products for the
256 extra factors through W3:

`2 * 1024 * 256 * (2 * 1024 - 1) = 1,073,217,536 FLOPs`.

Before vector/sort deltas, adding that to the measured cap-1792 total gives
`215,195,558,914`, below the inclusive 0.1-floor maximum
`219,902,325,555` by `4,706,766,641` FLOPs.  The actual decision is made only
by a retained whestbench 0.16 / flopscope 0.12 subprocess receipt.

The persistent factor-state increment at h2 is
`2 * 1024 * 256 * 4 + 256 * 4 = 2,098,176` bytes.  Temporary growth and
backend allocations make this an algebraic lower bound, so measured live RSS,
not this calculation, governs the memory gate.

## Gates

- total measured FLOPs no greater than `floor(2^41 / 10)`;
- sampled solution-process RSS below 8 GiB;
- residual time no greater than 0.4 seconds and wall time no greater than 120
  seconds;
- finite float32 output with shape `(16, 1024)`;
- no scientific accuracy claim from the 1,000-sample synthetic plumbing truth.

## TT decision

A conventional order-3 TT rank `r` uses `n * (r^2 + 2r)` core scalars and
the same number of dense-W action columns.  At `r=22`, that is 540,672
scalars/columns, already more than the 524,544 factor scalars plus 256 weights
needed by the exact 256-term symmetric-CP spill.  TT conversion/rounding adds
work and approximation.  Therefore a one-action TT is held: it is neither
cheaper nor scientifically separable from extra CP capacity here.  A future
TT test must persist for multiple layers and preregister a target-free
compression criterion to become a distinct hypothesis.
