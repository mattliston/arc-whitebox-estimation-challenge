# Current signed-pair query audit

September 6, 2026 PDT. Read-only audit against current bank SHA256
3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757.
No current-card run, target-bearing computation, change, or submission.

## Exact factor/query geometry

Current `_push_bank_shared_newborn` already shares the common first push
of each new signed pair. Later both dense factors are propagated; the
full 1024 output rows remain necessary for exact scalar k3 of every unit.
`plane` evaluates `(pp^2) @ (pp*lambda).T` only on active-by-active rows.
Final layer skips that dense plane and reads only cubics of the full push.

After writing P=H+D/2 and Q=H-D/2, the exact signed pair is
3*SymAverage(H,H,D)+D^3/4. Its repeated-plane evaluation can be arranged
as two N-width products rather than one 2N-width product, but that does
NOT halve the bilinear work. Both H and D must still be pushed. This
reparameterization by itself gives no >=25% whole-bank saving.

## Curvature-active-by-live query is already closed

The exact proposal to omit nearly-linear POSITIVE rows, while preserving
the full factor push and all scalar diagonals, was implemented in
`candidate_response_rows_cost_v2.py`, lines1626-1635. It selects
min(g,1-g) among alive coordinates, then reads

    left=pp[response_indices]; right=pp[live_indices]
    response_plane=(left*left) @ (right*lambda).T.

This is not the separate cold-state or factor-pruning branch. In the
exact positive-linear limit, the doubled endpoint's A derivatives of
order >=2 vanish, so an omitted K21 row does not affect covariance.
The target K21 contribution equals its g_i^2*g_j inherited CP term and
cancels from the star residual. But a fourth-moment/radial source survives:
the r=0 source sum is

    6 * sum_ij omitted_K21_ij * (g_i-host_i*d_i) * beta_j,
    beta_j=2*(gaussian_mean_j-host_j*g_j).

That implementation restores this source by two factor contractions,
subtracting the already-retained diagonal. It does not claim that finite
g>=.99 is exactly linear or that all higher-rho responses vanish.

The existing exact-linear certificate reports star-rest maximum error
1.1171619185e-15 and radial error1.3877787808e-15. The fixed matched
200-row/160-40/40-epoch round completed September5 by actual mtimes;
rank35/100 final raw changed +3.1204% / -0.3523%. The first exceeded its
predeclared3% gate. Earlier unlearned native saving was only5.85%, not
the requested25% whole-bank reduction. No reopening or threshold sweep.

Receipts: `RESPONSE_ROWS_MATCHED_ROUND_V1.md`,
`RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md`,
`response_rows_certificate_v1.json`, and `check_response_rows_v1.py`.

## Other exclusions checked

Shared difference-leg rank128/cuts8,12 cost only4.000%/4.357% net saving
after acquisition/restoration; old per-cohort SVD and plane-read SVD
variants are separate closed failures. The original/early single-cube
variants literally kept the plus atom and discarded the minus atom,
restoring only the current repeated plane; their poor scores do not
certify a newly solved pair coalescence, but also do not give it any
presumption of accuracy. Independent/fixed-count pair thinning, one-high
transport, symmetric triangle queries and both frozen/global-batch
Smirnov runtime failures remain closed.

Root separately authorized a tiny signed-pair FIRST-JET coalescence
certificate as a distinct approximation. Its own design/receipts must be
used, rather than labeling any previous failure a success or reopening a
closed mechanism by renaming it.
