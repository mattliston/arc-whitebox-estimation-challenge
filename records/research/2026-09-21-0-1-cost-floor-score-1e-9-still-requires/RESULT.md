# Remaining gap from the current official bank

The precise reported adjusted score is 3.50773885697508e-09, and reported
raw MSE is 1.9095373282596e-08. The archived evaluation records100 completed
networks and zero failures; it exposes50 rounded per-network score rows.
The requirements below use the precise official aggregate and do not infer
an exact all-network score decomposition from those rounded rows.

With identical predictions, the multiplier floor .1 gives adjusted score
at least 1.9095373282596e-09. Cost reduction alone therefore cannot reach the
reference target1e-9. Mean raw MSE must reach1e-8 or better even at the floor:
a reduction of at least 47.6313% from the reported raw MSE.
At unchanged costs, a uniform per-network MSE reduction would instead need
71.4916% to reach that point target.

For3e-9, unchanged predictions with uniform effective-cost scaling would
need a reduction of 14.4748%. Every recorded cost would remain
above the floor under that scaling, so the hypothetical proportional-score
calculation is valid. This is not evidence that such a cheaper implementation
exists, and it says nothing about its wall time or approximation error.

The current528-parameter covariance fit remains the sole numerical experiment.
No intermediate training metric was used to select a candidate or change its
schedule. All native, downstream and runtime gates remain unchanged.

Source review confirms that the earlier P fit already used the final-layer
objective. Prior Gaussian-suffix query fitting failed; exact single-query
suffix scheduling does not avoid the intermediate closed-loop trajectory.
Those scoped results are preserved, without treating them as universal
impossibility claims. No new experiment was selected from this review.

The bounded public review found a participant report on factorized K3 and
its failed alternatives. Its conjecture about leaders is not a reproduced
method or evidence for a new supplier:
https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218
