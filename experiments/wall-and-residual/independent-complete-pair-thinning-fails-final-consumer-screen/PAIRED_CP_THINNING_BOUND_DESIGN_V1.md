# Cancellation-preserving independent CP thinning: final-consumer screen

September 6, 2026 PDT. Phase2 only, fixed development ranks35/100, existing
bank P0 and scalar final Q. No labels, training, actual thinning, estimator
edits, new Monte Carlo truth or upload. This is not another SVD, cap/age
truncation, one-high tensor, or cohort-gain fit. It asks about independent
unbiased Bernoulli retention of COMPLETE signed birth pairs.

Each pair's final pre-skew contribution is evaluated stably as
a_j = lambda_j [3 u_j^2 d_j + d_j^3/4], with
u=(p_plus+p_minus)/2 and d=p_plus-p_minus. Keep the complete pair together.
For independent I_j~Bernoulli(p_j), its unbiased estimate is I_j*a_j/p_j.

All incoming states, covariance, star/trace and readout messages are fixed
to the original bank. The scalar final map's Jacobian with respect to k3
is diagonal plus rank one, J=diag(direct)+global_response*b.T. This holds
because only final feature29 couples current scalar marginals across units;
all messages come from the fixed incoming state and gates do not depend on
k3 at fixed mu/v. Here b_i = d(host_i)/d(k3_i)/(n*sigma_i).
Preserve the final nonnegative output clipping in the derivative.

For s_j^2=||J*a_j||^2/n the exact linearized output-MSE variance is
sum_j (1/p_j-1)*s_j^2. Subject to sum p_j=M, its best independent-Bernoulli
probabilities satisfy p_j=min(1,s_j/tau). Use fixed80 bisection iterations.
The declared M is exactly half the pairs. Oracle probabilities may inspect
every existing final pair and the frozen readout derivative; acquisition,
earlier-layer noise, state feedback, and all costs are FREE. This is an
optimistic necessary screen for this frozen-consumer sampling proposal,
not a universal bound on deterministic compression or coadapted models.

Certificates before a wide claim:

- A tiny explicit enumeration of all independent retention patterns must
  agree with the variance formula to1e-11 relative.
- Pair sum must agree with the original signed-cube final k3 sum to1e-10
  relative in f64 arithmetic on the same stored float32 factors.
- The diagonal-plus-rank-one final Jacobian must agree with full reverse
  derivatives for one fixed target-blind cotangent to1e-5 relative.
- Repaired MPS full baseline remains within the existing2e-6 max-mean gate
  against the archived NumPy reference on each net; no fallback.

The sole screening gate is best linearized added final-MSE <=1e-9 on BOTH
nets at half expected pair count. This allows roughly5% of current raw
error while imagining a substantial CP saving. Failure closes this exact
independent-pair thinning proposal before draws, a carrier, probability/
budget sweeps, coadaptation, or panel work. Passing only permits a separate
whole-loop cost/error design; it is not a score improvement.

Use one nice10 worker, at most3 CPU threads and one MPS job. Record actual
clock, hashes, process/GPU memory qualifications and no-target access.
