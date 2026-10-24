# Q24 manual rho-power recurrence successor

Status: **preregistered target-free source-only development; immutable HOLD is
the strongest authorized outcome.  No full-shape meter, target, truth, score,
dataset, fit, sample, package, submission, or canonical-graph mutation is
authorized.**

## Question and immutable parent

Can the frozen Q24 closure replace only the slow axis-0 cumulative-product
spelling of `rho^1..rho^24` with an association-preserving plane-wise
recurrence into its existing native-output arena, while keeping bitwise Q24
science and the exact Phase-2 FLOP bill?

The sole parent is
`../fullbudget_k21_k4_closure_q24_v1/estimator.py`, 68,115 bytes, SHA-256
`b38cd865297c344626da3c4f414e1d4923a9efdc62147e3103f155b2034f7b52`.
Its final freeze receipt is SHA-256
`565093c418a656510cc1ac15b6d93cd50b104a122bf2d71586c41e32f693026d`.
The parent's literal target-free meter is authoritative at
`2,131,249,329,109` FLOPs, 6,878 operation calls, 28.354738 backend seconds,
29.222195 wall seconds, 0.294491 residual seconds, and 4,507,058,176 bytes
descendant RSS.  Those measured times and bytes remain parent-only evidence.

## Frozen single change

Inside `_q24_context`, retain the existing `powers[0] = 1` native copy.  Replace
only

```python
fnp.cumprod(
    fnp.broadcast_to(correlation, (24, width, width)),
    axis=0,
    out=powers[1:],
)
```

with this exact left-associated recurrence:

```python
fnp.copyto(powers[1], correlation)
for power in range(2, _Q24_TERMS):
    fnp.multiply(powers[power - 1], correlation, out=powers[power])
```

Thus plane 1 is exactly `rho`; for every integer `q=2..24`, plane `q` is
computed as the immediately preceding stored plane times the same `rho`.
There is no exponentiation, reassociation, dtype change, new arena, temporary,
conditional, or change to the inclusive `q=0..24` consumer axis.  All CP3,
H3, CP4, H4+alpha, L4, `L3^2/2`, PSD repair, W16 specialization, admission
guard, 17 derivative channels, coefficients, constants, and remaining source
bytes are frozen to the parent.

## Numerical and billing authority

The reference is the immutable parent's pinned FlopScope 0.12 `cumprod` with
native `out`.  Strict equality means NumPy `array_equal`/byte-identical output;
no tolerance may rescue a mismatch.

Pinned FlopScope accounting bills an axis-0 accumulation over shape
`(24,w,w)` as `(24-1)w^2 = 23w^2` multiplications.  The replacement has exactly
23 pointwise multiplies of shape `(w,w)`; `broadcast_to` and `copyto` are
zero-FLOP data-movement calls.  Therefore the registered FLOP delta is exactly
zero per context and per complete depth-16 prediction.  The expected call
delta is `+22` per Q24 context: parent `broadcast_to+cumprod` becomes one
additional `copyto` plus 23 `multiply` calls.  With 15 Q24 contexts, the
complete prediction expectation is `+330` calls, from 6,878 to 7,208, while
the projected production bill remains `2,131,249,329,109` FLOPs.  Tiny meters
must verify these identities; no production/full-shape execution is allowed.

## Frozen target-free gates

1. **Source/static:** exact parent hash; exact import set; no forbidden import,
   target, reader, fit, sample, score, submission, worker, cache, dynamic code,
   or file/network operation; production width/depth/budget guard unchanged;
   a whole-module AST comparison must prove that `_q24_context` is the only
   changed executable function and that its replacement matches the frozen
   recurrence above.
2. **Tiny Q24 exact/bitwise:** fixed width-4 controlled coefficient, sigma,
   and covariance cases covering zero, signed correlation, diagonal one,
   values near one, and repeated powers must produce all 25 rho planes and
   every returned context field bitwise equal to the parent.  Candidate q0
   must be exactly one and q1 exactly the clipped correlation.
3. **Poisoned arena:** before each focused context run, fill the complete
   25-plane parent and candidate rho arenas with NaNs.  All planes must be
   overwritten, finite on finite fixtures, alias their persistent arena, and
   remain bitwise parent-equal.  Any surviving poison rejects the source.
4. **Four canonical depth-16 fixtures:** at width 8, run exactly identity
   float32; near-identity seed 20260823 scale 0.02 float32; QR seed 20260823
   scale 0.35 float64; and pure-normal seed 17 scale 0.12 float32.  Completion,
   finiteness, every reported `[16,8]` value, and output digest must be bitwise
   parent-equal.
5. **Official-He small panels:** run exactly `(width,seed)` = `(32,20260825)`,
   `(32,20260826)`, `(64,20260825)`, `(64,20260826)`, `(128,20260825)`, and
   `(128,20260826)`, depth 16, with float32 weights generated as
   `default_rng(seed).standard_normal((w,w))*sqrt(2/w)`.  Compare parent and
   candidate under the same exception-catching law.  Completed outputs must
   be finite and bitwise equal; failed panels must match exception type and
   message.  Existing off-contract nonfiniteness remains a negative control,
   not a production-science claim.
6. **Exact tiny price:** whole-prediction identity fixtures at widths
   `4,5,6,8,12,16` must show exactly zero FLOP delta and exactly +330 calls.
   The isolated rho kernel must independently show `23w^2` FLOPs in each form
   and +22 calls in the candidate spelling.  Algebra and tiny receipts may
   project the unchanged production bill, but only a separately authorized
   full-shape literal meter could measure candidate wall, residual, memory, or
   production call telemetry.
7. **Reproducibility:** run the complete target-free suite twice after the
   estimator bytes freeze; deterministic scientific fields and digests must
   match.  Timestamps or observed wall diagnostics are non-authoritative.

## Outcome law

- If every gate passes, freeze one additive estimator and evidence package as
  `IMMUTABLE_HOLD__METER_ONLY_AUTHORITY`.  It may be handed to an independent
  auditor, but it is not package-ready, meter-qualified, or submission-ready.
- Any source, bitwise, poison, fixture, official-He outcome-parity, or exact
  billing mismatch rejects these bytes cleanly.  Do not tune against the
  result in this namespace.
- Even on pass, do not claim a wall improvement.  The replacement is only a
  source-equivalent wall hypothesis until a separate authority preregisters
  and runs a full-shape meter.

## Access and resource envelope

- Legal: frozen source/manifests/receipts, static inspection, synthetic
  weights-only fixtures, pinned FlopScope, small CPU panels, additive files in
  this experiment directory.
- Forbidden: all datasets and targets (public or synthetic), truth generation,
  accuracy/raw-MSE/score computation, full width 1024 execution, packaging,
  board/API submission, network writes, and canonical graph edits.
- Counters begin at zero and must remain: target reads 0, datasets 0, truth 0,
  fits 0, samples 0, full-shape runs 0, packages 0, submissions 0, graph edits
  0.

This experiment cannot establish raw MSE, accuracy, board behavior, wall-time
improvement, production memory, or deployability.
