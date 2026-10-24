# Q24 two-point marginal-moment repair probe

Status: **preregistered target-free source-independent development probe**.
No target, truth, score, dataset row, full-width execution, estimator freeze,
package, submission, or graph mutation is authorized by this file.

## Fixed parent and question

The only parent is the frozen manual-rho Q24 estimator, 68,143 bytes, SHA-256
`7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00`.
Its source-only independent audit is SHA-256
`b583911142d523e16ee2208289a8b79137669d256e45432877cf3bdf3785efc6`.

The certified V4 diagnostic found 9 represented marginal tuples with negative
`A = r1*r3-r2^2` and 13 with negative
`B = (r4-r2^2)*(r2-r1^2)-(r3-r1*r2)^2`, concentrated in near-dead late-layer
units.  This probe asks only whether a mathematically coherent marginal
completion removes those failure modes on fixed target-free stress fixtures.

## Frozen intervention

Immediately after `_univariate_edgeworth`, compute represented-value `A` and
`B` with the parent association above.  A unit is invalid iff `r1 <= 0`,
`A < 0`, or `B < 0`; there is no fitted epsilon.

For an invalid unit with `r1 > 0`, retain `r1,r2` exactly and replace the pair
`(r3,r4)` jointly by

```
a   = r2 / r1
r3' = r2 * a
r4' = r3' * a
```

These are the moments of the unique two-point law supported on `{0,a}` with
mass `r1^2/r2` at `a`; it matches `r1,r2` and lies on both Stieltjes boundaries
in exact arithmetic.  For `r1 <= 0`, use the degenerate law
`r1'=r2'=r3'=r4'=0`; the parent's existing covariance variance floor remains
the downstream numerical regularizer.  Valid units remain byte-identical.

This is a joint marginal completion, not an independently tuned raw3/raw4
clamp.  It does not repair joint pair moments and therefore cannot establish a
joint law or accuracy.

## Fixed fixtures and outcome law

Run both the unchanged parent and the in-memory repaired seam through internal
`_predict` under FlopScope, with no targets:

1. width-8 depth-16 identity float32;
2. width-8 near-identity, RNG seed 20260823, scale 0.02 float32;
3. width-8 QR, RNG seed 20260823, `0.35*Q` float64;
4. width-8 pure-normal, RNG seed 17, scale 0.12 float32;
5. official-He law at `(width,seed)` equal to `(32,20260825)`,
   `(32,20260826)`, `(64,20260825)`, `(64,20260826)`,
   `(128,20260825)`, `(128,20260826)`.

For every arm record completion/exception, finiteness, output digest, repair
counts by reset, and repaired post-sign counts.  For finite pairs record final
and all-layer disagreement MSE and maximum absolute difference.

- **Mechanism pass:** every repaired tuple has represented `A>=0` and `B>=0`,
  no parent-finite fixture becomes nonfinite, and at least one registered
  parent failure becomes finite or a registered negative tuple is repaired.
- **Mechanism reject:** any repaired post-tuple is negative, any parent-finite
  fixture becomes nonfinite, or no intervention ever activates.
- **Scientific hold regardless:** a pass authorizes only construction of a
  separately frozen source candidate and broader target-free gates.  It does
  not authorize width 1024, raw MSE, packaging, or submission.

Public submission #328210's five-row tail concentration motivated the question
but does not set a threshold, fixture, coefficient, or outcome gate here.

