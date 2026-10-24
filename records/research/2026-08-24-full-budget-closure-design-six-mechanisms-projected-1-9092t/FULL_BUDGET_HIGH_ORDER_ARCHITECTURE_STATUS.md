# Full-budget high-order closure architecture status

Status: **active derivation, target-free, not yet a frozen estimator or
scientific accuracy result**.  This note keeps the live graph synchronized
with the current closure-only program.

## Exact foothold

For the standard Gaussian input, `z1` is jointly Gaussian with zero mean and
covariance `W1.T @ W1`; its third and fourth cumulants are exactly zero.  The
Gaussian-ReLU law therefore gives the full `h1` mean and covariance exactly,
and the affine contractions into the `z2` mean and covariance are exact.  The
post-ReLU law at `h2` is the first closure step; no higher cumulant of `h1` or
output at `h2` is called exact.

## Active state

The current leading state is:

- dense mean and covariance;
- a full-birth, cap-16384 symmetric CP3 carrier, read into marginal `K3` and
  both orientations of the mixed `K21` action;
- a coherent quartic action `F4 = Star4(H) + alpha P_C`, reset from the same
  corrected raw moments used for mean, covariance, `K21`, and `K22`; and
- a correlation-space eigenspectrum projection followed by exact diagonal
  renormalization, never the failed global off-diagonal shrink.

The quartic gauge is provisionally

```text
H_ii = d_i - 3 alpha C_ii^2
H_ij = A_ij - 3 alpha C_ii C_ij
S_ij = C_ii C_jj + 2 C_ij^2
alpha = <B,S> / <S,S>       (strict upper triangle)
```

This spelling retains `K31` coherently instead of omitting it or scheduling
the previously damaging synthetic/product `K31` plane independently.  It
also supplies the `K22`, `211`, and `1111` actions through `P_C`.  The choice
is forced by two pieces of preserved evidence: the oracle bivariate order-four
consumer needs `k3`, `K21`, `k4`, `K31`, and `K22`, while the diagonal-only
CP4 sidecar recovered 77.89% of the oracle mean gain but failed its mixed-plane
and PSD gates.

## Provisional full-budget envelope

The current dtype-aware conservative ledger is approximately `1.9092T` FLOPs,
about 86.82% of `2^41`, leaving roughly `289.82B`.  Its dominant blocks are the
measured `620.091B` CP16384 host, `450.957B` fused K21 plane reads through z15,
`122.347B` cubic residual star, `182.447B` coherent quartic star, `330.683B`
for fourteen float64 spectral repairs, and approximately `196.478B` for the
Q32 bivariate consumer through `L4 + 1/2 L3^2`.  W16 is specialized exactly
to the marginal mean action, so it does not materialize unused final K21/K31/
K22 or covariance planes.  Weight handling, connected-moment resets, and
elementwise assembly receive an additional `6B` conservative allowance.

The complete cost-reopen classification is recorded in
`FULL_BUDGET_COST_REOPEN_AUDIT.md`.  In particular, dense exact K3 transport
remains impossible at width 1024, while the previously cost-screened
structured state and PSD repair are now affordable.

These numbers are a design boundary, not a resource PASS.  Promotion requires
literal FlopScope accounting, finite `[16,1024]` output, residual time below
0.4 seconds, solution memory below 8 GiB, and an independent source/resource
audit.

## TT boundary

The currently preserved TT24/s28 route costs about `1.463T` for fifteen pushes
before the CP3/K21 host and also failed its own 0.15 action gate on the standard
source-blind fixture.  It remains a disabled seam, not part of the leading
estimator.  A TT successor must either replace—not duplicate—the coherent
quartic action or prove a materially better action/cost envelope.

## Accuracy boundary

The measured pure-closure leader remains the full-birth recursive-mean CP16384
estimator at final raw MSE `9.340196551564291e-7`.  No target has been opened
for the architecture described here.  Its purpose is to repair the first
missing joint state beginning at layer 3 and drive the closure itself into the
`1e-8` class before any Kerdock, sampling, learned, or fitted residual is added.
