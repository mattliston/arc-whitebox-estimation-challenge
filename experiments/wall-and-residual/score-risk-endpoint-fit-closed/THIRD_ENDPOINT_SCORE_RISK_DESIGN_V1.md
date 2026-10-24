# Same endpoint features, independently justified score-risk objective

September 6, 2026, Phase2. The completed fixed-field risk audit establishes
that a mean-direction Gaussian tilt predicts actual fresh-weight response
energies to about 4--6%, whereas the old covariance/K21 loss substantially
overstates useful error removal. This is positive independent mathematical
evidence for an objective change, not an iteration retry of the failed
K21 ridge system. Its existing optimum bound remains valid for that system.

One fixed TRAIN cell mini57/pre13 only. Reuse exactly the old input-only
H65 endpoint span, correlation powers 0/1/2, full-strength third-plane
target, and exact linear K21-to-covariance kernel K. No new data, basis,
features, rank, marginal write, correction strength, or state trajectory.
The objective uses only post13 bank mean/covariance for its metric, not
actual W14 columns, actual next response, or either target in its geometry.

Let sc=sqrt(2 mean(K^2)), an input-only normalization giving unit average
covariance energy for independent unit-variance standardized plane entries.
Let B(theta)=[K*S(theta)+(K*S(theta))^T]/sc and Y=target_covariance/sc.
For A=I-beta uu^T certified in the preceding audit, fit

  J(theta) = [tr(A E A E) + .5 tr(AE)^2]/(2 n^2)
             + .001 ||theta||^2/2,  E=B(theta)-Y.

Its metric action is M(E)=A E A + .5 tr(AE) A, evaluated by rank-one
updates. This is the certified Gaussian risk up to a positive constant.
Keep ridge .001 in the newly dimensionless covariance units; it is NOT
claimed to impose the same prior as the differently normalized old loss.
Zero-start PCG, tolerance1e-7, maximum1000 iterations fixed before results;
diagonal preconditioner 2 H^2.T[(K/sc*Rpower)^2]H^2/n^2+.001,
which deliberately ignores transpose cross terms and the tilt, but does
not alter the system. Small dense solve/adjoint/metric certificates first.

Require converged true normal residual, analytic risk residual ratio <=.50
AND actual W14 next-mean-response residual ratio <=.50 at full strength.
Report ordinary covariance and old weighted-plane ratios without selecting
on them. Also compare the preceding fitted field's risk as a fixed control.
No alternate solve budget, ridge, metric, objective, rank or seed sweep.

Failure closes this fixed score-risk endpoint span. A pass permits a
separately designed cross-network learnability test, not automatic access
to the other seven preselected public files or a carrier/score submission.
This is a regularized in-sample oracle coefficient test, not an
unregularized information ceiling or a generalizing estimator.
