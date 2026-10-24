# No-refit bound on the unfinished score-risk ridge solve

The fixed 1000-step fit failed both response gates (.772535 analytic,
.801489 actual W14), and its normal residual4.2066e-7 misses tolerance1e-7.
Do not add iterations or relax any gate. Determine whether an exact solve
of THIS SAME ridge system can possibly pass the analytic .50 gate.

Write T=M^(1/2) B/n and normal N=T^T T+lambda I. For computed residual r,
the exact solution displacement satisfies

  ||T delta_theta|| <= ||r||/(2 sqrt(lambda)),

because max_{s>=0} s/(s^2+lambda)=1/(2sqrt(lambda)). Physical risk is
2*factor*sc^2*n^2 times squared transformed norm. Hence use the triangle
inequality to lower-bound the exact optimum's target-residual risk from
the observed risk norm minus this computable displacement bound.
Inflate the displacement bound by1.001 as a numerical margin; this is
not an interval certificate or a statistical confidence bound.

One normal application only, no optimization or coefficient changes.
The bound concerns only this fixed regularizer, features and risk system;
it is not an unregularized capacity ceiling or a bound on other models.
