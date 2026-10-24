# Historical training and moment-noise evidence

The older terminal80 run adds two 160-network passes to its old P1 host.
Its original training qualification remains failed. A separate inference
qualification recovered all40 outputs plus repeat from the sole fixed
terminal payload, reproducing the13 historical validation outputs exactly.
The saved-output magnitude screen compared terminal MPS predictions with
P1 native predictions on the same1B means: raw ratio0.998086518 and23/40
wins. It failed its fixed ratio<=0.90 and>=24-wins gate. This is a mixed
backend screen, not a negative native adjusted score or a convergence bound
for the current host. No current extra-epoch experiment is selected here.

All eight incoming-moment payload hashes were rechecked, and their ZIP
directories were inspected without opening member streams or decoding
arrays. They contain pooled moments through fourth order and a single
sample count/seed; they contain no sample blocks, independent replicate
moments, or sixth/eighth-order moments. The previous joint-head validation
next-mean ratios0.917359 and0.931170 remain failures of its fixed0.90 gate.
Its Gaussian first-layer and FP32 rounding checks do not quantify late-layer
Monte Carlo noise. Its own target receipts explicitly say that this noise
was not removed.

An exact rational counterexample illustrates the missing information. Two
symmetric scalar laws can both have mean0, variance1, third moment0 and
fourth moment3, yet have different eighth moments and different sampling
variances for the fourth cumulant. Here the cumulant influence function is
X^4-6X^2+3; its variance is18 for one law and10 for the other. Thus even a
scalar fourth-cumulant variance is not identified by these aggregates.
The actual transported joint consumer additionally needs covariances across
its moment entries. A late-layer Gaussian approximation would be an extra
model assumption, not measured noise correction.

This audit neither estimates that missing noise nor changes either failed
gate. Establishing it would require additional appropriate sample statistics
or independently acquired moments under a separately specified study. No
samples, numerical backend, estimator, fit, candidate, or target array ran
or changed here. The currently running downstream validation is unaffected.
