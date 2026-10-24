# Coherent state scale: full12-coefficient local-response test

2026-09-05,14:12PDT. Declared before scoring targets for this action.
Run ONLY after the algebra, exact NumPy zero-action bank parity, actual
CPU secant match and local-branch derivative gates pass on35/100.

Fit-side networks:35,36,38,40. New-update holdouts:37,39,42,48.
They are excluded from this new solve, not claimed unused by every old
experiment or parent model. Existing documented-1e9 final post means are
the reference; the npz files do not prove the sampler/count provenance.
No new Monte Carlo labels, private targets or forum posts.

Compute the COMPLETE12-column final-output Jacobian at zero action on all
four fit-side networks, through all CP/star/trace, covariance and frozen
learned-head updates. Save every zero baseline and derivative. This is a
local selection-branch derivative, not a globally smooth model. Real
forward evaluations keep dynamic selections.

With pooled response J and parent error e, solve the single convex problem

    min_theta mean((e + J theta)^2)
              + .001 * sum_j mean(J_j^2) * theta_j^2
    subject to ||theta||_2 <= .25.

Use a Lagrange-multiplier bisection if the unconstrained solution lies
outside the radius. Report response conditioning and active constraint;
do not select regularization/radius/features/networks after seeing results.
This is a complete small response-span solve rather than one gradient
direction. It is not the earlier conditional-birth proposal reopened.

If predicted mean raw improvement on the fit side is below20%, stop.
Otherwise compute ONE forward derivative in the selected theta direction
on each of the four holdouts, without retuning. Require predicted mean
raw reduction>=20% AND reduction>=10% on every holdout. Only a pass
warrants independent nonlinear NumPy evaluation at the same theta, which
must meet those same thresholds before a larger dataset/refinement is
designed. No long fit, parameter sweep or panel merely for a marginal gain.

This is not a deployable score or complexity claim. Any actual promising
update still needs honest broader validation, both14-net panels, metered
cost, memory/residual/wall timing and packaging parity before submission.
One nice10 CPU worker,3threads,15-minute deadline and immutable receipts.
