# Q24 certified connected-carrier shutdown v2

Status before source mutation or execution: `PREREGISTERED_TARGET_FREE_SOURCE_PROBE_ONLY`.

Parent is the frozen manual-rho Q24 estimator SHA-256
`7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00`.
This spelling supersedes incomplete v1 source SHA-256
`2532e58b76fd839d699b089c11e729d958f33edc1499816e9c75ac590b6aef2c`,
whose checked target arrays were zero but whose nonzero global pairing channel
reintroduced invalid-incident stored K22.  The v1 erratum and independent
red-team are bound by SHA-256
`50b81022490dc81c989a6db9d9d066cb3244c34efc9e33be05ce01b691660e6b` and
`79c4087923e4d18b41420203c3a1bd93e95af3ab5634058ea97357b4ef19aee0`.

## Frozen mechanism

At every recursive reset, compute the same outward-rounded represented-tuple
Stieltjes certificate as construction probe v1.  For each certified-invalid
unit, retain the reported mean and lower-order covariance state, while:

- zeroing every incident K21, K31, and K22 target and its K4 target;
- masking the CP3/CP4 post-ReLU gate and CP3 birth coefficient; and
- if *any* unit is invalid at that reset, forcing the global scalar
  `pairing_alpha` to represented `+0.0` before quartic residual construction.

The last rule closes the missed carrier seam: CP4 is incident-zero under the
masked gate and coordinate birth, the cubic and quartic stars are then
incident-zero, and no global covariance-pairing K22 term can reappear there.
Valid-valid K22 residual carried only by that scalar is deliberately dropped
on an affected layer.  No epsilon, magnitude threshold, layer threshold,
target feedback, retry, or fixture-specific branch is allowed.

## Frozen target-free roster and gates

Use the exact ten fixtures from helper SHA-256
`2c37c363155ac2b709de00a0d7a949a33a43106592e0df88daa7509bf2468b2d`.
The source probe passes only if:

1. at least one shutdown activates;
2. no parent-finite fixture becomes nonfinite or raises;
3. identity and near-identity, on which no shutdown activates, are byte-identical
   to the parent;
4. every affected reset returns pairing_alpha exactly zero and has no nonzero
   cubic/quartic-star value incident to an invalid coordinate;
5. all finite-pair disagreements satisfy both
   `RMS(child-parent) <= RMS(parent)` and
   `maxabs(child-parent) <= maxabs(parent)` (zero parent scale requires exact
   zero disagreement); and
6. static source diff, legal imports, guard, and exact small-width resource law
   pass.

A pass authorizes one independently audited immutable source HOLD and then a
separately released width-1024 target-free comparator/meter.  It does not
authorize targets, scoring, packaging, submission, graph promotion, or an
accuracy claim.  Failure closes this exact spelling without threshold tuning.
