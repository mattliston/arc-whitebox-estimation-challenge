# Conditional birth: one shared-gradient information test

2026-09-05, approximately 13:52 PDT. Declared before reading new gradients.
This is the 192-coefficient, quadratic-depth, current-state-conditioned
signed-birth action, with exact stored-coefficient plane compensation.
It is not another fit of the already closed uniform birth amplitudes or
existing output heads. All existing learned parameters stay frozen.

Use ranks 35..58 excluding 37,39,42,48 as 20 fit-side networks; those four
are held out of this new update only, not claimed unseen by every previous
experiment. References are the existing documented-1e9 final means. Their
files do not prove the underlying sampler/count provenance.

1. At zero action, compute the full CPU reverse derivative of each fit-side
   final MSE / 2e-8. Use the unit direction opposite their mean gradient;
   no per-network or validation-dependent direction choices.
2. Compute full forward-mode responses in this direction on the fit side.
   Select ONE nonnegative step by minimizing the pooled linear-response
   quadratic, restricted to Frobenius parameter radius .25. No step sweep.
   If even the predicted fit-side raw reduction is below 10%, stop.
3. Only otherwise compute the four holdout directional responses, without
   changing the direction or step. Require predicted mean raw reduction
   at least 10%, and at least 5% reduction on every holdout.
4. Only a passing information test warrants independent actual NumPy
   execution at that single step. Require the same holdout thresholds
   before designing a longer conditional-birth fit. No epoch/capacity/
   radius/direction shopping after a failed gate in this fixed round.

All forward executions retain actual dynamic selections. The derivative
is a certified local-branch derivative, not a prediction that selection
boundaries disappear. Linear-response improvements are not scored gains.
No submission, billing or runtime claim is permitted by this diagnostic.
One nice-10 CPU worker, three threads, a 15-minute limit and immutable
per-network progress receipts. No MPS, forum writes or sealed targets.
