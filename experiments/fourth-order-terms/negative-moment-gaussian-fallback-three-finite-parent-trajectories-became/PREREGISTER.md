# Q24 negative-moment Gaussian fallback probe

Status: **preregistered target-free development probe**.  No target, truth,
score, dataset row, full-width run, package, submission, or graph mutation is
authorized.

The frozen parent is manual-rho Q24 SHA-256
`7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00`.
The prior two-point boundary repair is rejected by receipt SHA-256
`5b9af5b607b9102d2140bc92a643e9ab626c35e0ff7bead530d6e8abb247a519`
because it activated 4,741 times, introduced four finiteness regressions, and
recovered no failure.

## Frozen intervention

At the output of `_univariate_edgeworth`, a nonnegative random variable cannot
have a negative raw third or fourth moment.  Define a unit as invalid exactly
when represented `corrected3 < 0` or `corrected4 < 0`; use no epsilon and do
not inspect A/B cancellation.  For an invalid unit, replace the complete
corrected `(r1,r2,r3,r4)` tuple by the already-computed zeroth-order Gaussian
positive-part tuple `(raw1,raw2,raw3,raw4)` from the same pre-mean and
pre-variance.  Valid units remain byte-identical.

This is a coherent local distribution fallback rather than four independent
floors: the replacement tuple is exactly the positive-part Gaussian host that
the Edgeworth expansion corrects.  It intentionally discards all univariate
high-order corrections for an incontrovertibly invalid unit.  It does not
repair joint off-diagonal moments and proves no accuracy.

## Frozen gates

Use the same ten fixtures and exact weight construction frozen in
`../q24_two_point_moment_repair_probe_v1/PREREGISTER.md`: four canonical
width-8 fixtures and the six official-He `(width,seed)` pairs at widths
32/64/128 and seeds 20260825/20260826.  Run parent and fallback separately
under FlopScope, with no targets.

Record completion, finiteness, activation counts, output hashes, and finite
pair disagreement.

- Pass this construction gate only if it activates at least once, every
  returned third/fourth moment is represented nonnegative, and no parent-finite
  fixture becomes nonfinite.
- Reject on any parent-finite regression, any negative returned r3/r4, or zero
  activation.
- Recovery of an existing parent failure is supporting evidence, not required.
- Regardless of pass, width 1024, raw MSE, source freeze, package, and board
  promotion remain separately held.

The public five-row tail is motivation only and does not define this trigger or
any threshold.

