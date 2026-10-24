# K42+K51 source-side scalar-witness counterfactual

Status: `HOLD_USER_SUPERSESSION_REQUIRED__COUNTERFACTUAL_FORBIDDEN_RETRY_NO_RELEASE`.

This package is a static counterfactual accounting result, not a live graph
child and not authority to repeat the terminal K42+K51 experiment.  It opens
no row, weight, name, target, score, release, rank payload, estimator, or
submission path.

## 1. Terminal parent and governance

The immutable parent transaction is bound by:

- terminal closure receipt SHA-256
  `3334080f68af40fa02509639e7c9a6ebc87c8888c30b21aea399d9362bae793d`;
- frozen V3 source SHA-256
  `547b45c219fd868d359ebb8deec9f564a89d433cd6fbdc9cc637940e164cd601`;
- frozen V3 runner SHA-256
  `fd9a5f8f395f96af3569d7f2d422033bcd714c18d8b22760bb5e1fce0c9fbc8e`;
- frozen V3 map SHA-256
  `2248c8dc67cceabdef1f1228a4d77acfd7b5d9aecd250dfb162a9e19b57a5b2a`;
- reconciled canonical graph SHA-256
  `84ff484740ec9390d61272e9807784396db9d944abcc84b2d2490e4e23f47e76`.

The parent ran exactly once at observer-fit rank 201.  It passed 33 of 34
gates and closed on the sole frozen resource-residual gate because B measured
`0.5109498272649944 >= 0.4` seconds.  Its durable claim forbids retry and rank
643.  The graph marks `P2R0C1D29-K42K51-S0` `closed_measured`,
resource-residual-only, with no successor.  Therefore the present rewrite is
counterfactual evidence only.  No deterministic fresh-rank law is defined or
evaluated: selecting a rank would itself misrepresent the terminal authority.

Only explicit user supersession could authorize a new, separately reviewed
program outside this package.  This package remains incapable of creating a
release even after such a decision.

## 2. Scientific path left unchanged

The inherited scientific path is still the exact D4-Q7 centered construction:

1. form the same `d4 x d2` and `d5 x d1` Q7 sources;
2. recover connected oriented `K42` and `K51` with the same coefficients and
   the same two legal `fill_diagonal` sites (30 runtime calls);
3. apply the same `W4/W2` and `W5/W1` batched actions with factors 15 and 6;
4. form the existing `raw_total = raw_by_sector[0] + raw_by_sector[1]` once;
5. expose the same 15-by-1024 combined raw-k6 coordinate bank without an
   endpoint, reset, injection, target, or autonomous consumer.

The instrumentation is read-only.  It has no `out=` destination pointing at a
source plane, sector action, or `raw_total`.  The one extra vector-valued
operation writes a fresh duplicate sum.  Thus K42 bytes, K51 bytes, both
sector-action bytes, the existing `raw_total` bytes, and the final coordinate
bank bytes are invariant under the rewrite.  This is a static non-mutation
claim; no new production byte hash or runtime result is claimed.

The scientific spelling is dormant evidence, not an executable source path.
`_load_base`, `_install`, and `predict_with_pair_tape` have no decorators or
default-expression side effects and each begins, after an optional docstring,
with the same direct unconditional
`raise RuntimeError("COUNTERFACTUAL_FORBIDDEN_RETRY_NO_RELEASE")`.
Consequently the D4 module cannot be imported, a base object cannot be
installed, and neither internal nor public prediction can be reached.  The
static verifier proves this precedence directly from the source AST.

## 3. Lean scalar witnesses

At each of the 15 source hooks, while `planes` is still a binary64
`flopscope.numpy` array with shape `[2,1024,1024]`, compute:

```text
plane_min = fnp.min(planes, axis=(1,2))
plane_max = fnp.max(planes, axis=(1,2))
diag_nz   = fnp.count_nonzero(fnp.diagonal(planes, axis1=1, axis2=2))
```

Only the two minimum scalars, two maximum scalars, exact dtype/shape metadata,
and the single diagonal count pass to the sink.  A full K42 or K51 plane never
reaches plain NumPy.

After the unchanged action and factor scalings, compute in the same source
hook:

```text
sector_min = fnp.min(raw_by_sector, axis=1)
sector_max = fnp.max(raw_by_sector, axis=1)
duplicate  = fnp.add(raw_by_sector[0], raw_by_sector[1])
sum_equal  = fnp.array_equal(duplicate.view(fnp.uint64),
                             raw_total.view(fnp.uint64))
```

The uint64 view makes the duplicate-sum witness byte-exact, including signed
zero and NaN payload distinctions.  Only `raw_total`, already required for
the preserved coordinate bank, is copied to the local sink.

For nonempty real arrays under NumPy min/max semantics:

- all values are finite iff both extrema are finite, because NaN propagates
  and either infinity appears in at least one extremum;
- after finiteness is established, the array is nonzero iff `min < 0` or
  `max > 0`;
- the two plane diagonals are exactly zero iff their combined nonzero count is
  zero;
- the existing combined vector equals the two-sector sum byte for byte iff
  the uint64 `array_equal` witness is true.

These are equivalent to the parent hard finite/nonzero/diagonal/exact-sum
gates.  They are not materiality thresholds and cannot select a rank.

## 4. Diagnostics honestly removed

The counterfactual does not compute or promise any source-plane Frobenius
norm, source-plane RMS, source-plane max-absolute value, plane content hash,
action L2 norm, action RMS, action max-absolute value, sign fraction, zero
fraction, or sector cosine.  The signed min/max scalars are gate witnesses,
not replacements for those descriptive diagnostics.  No transient C42/C51
central-moment diagnostic is promised.

## 5. Exact official accounting

Let `n=1024`.  Per source layer the added official bill is:

```text
plane min+max     = 8*(n^2 - 1)
sector min+max    = 8*(n - 1)
diagonal count    = 4*n
duplicate f64 add = 2*n
uint64 equality   = 2*n
total per layer   = 8*n^2 + 16*n - 16
```

Across 15 layers:

```text
15*(8*n^2 + 16*n - 16) = 126,074,640 FLOPs.
```

Adding this to the independently confirmed integrated parent increment gives:

```text
151,825,424,384 + 126,074,640 = 151,951,499,024 FLOPs.
```

The existing hard cap remains `152,200,000,000`, leaving `248,500,976`
FLOPs.  The counterfactual B total is `1,510,469,538,959`, leaving
`688,553,716,593` below `2^41`.  The standalone-dense diagnostic would become
`152,150,195,984`; it is not a runtime equality gate.

Exact added operation calls are:

```text
min 30, max 30, diagonal 15, count_nonzero 15,
add 15, array_equal 15.
```

The complete predicted B-minus-A operation-call delta is the parent inventory
with `add=180`, `diagonal=30`, and the new `min=30`, `max=30`,
`count_nonzero=15`, and `array_equal=15`; every other parent count is
unchanged.

The largest remote witness allocation is one fresh binary64 duplicate vector,
`1024*8 = 8,192` bytes.  On the client, `finalize()` allocates the final
15-by-1024 bank while all 15 separately captured combined vectors remain live.
Those two simultaneous banks are `122,880 + 122,880 = 245,760` bytes.  Added
to the parent scientific projection `7,043,162,112`, the counterfactual
process projection is `7,043,407,872 < 8 GiB`, leaving `1,546,526,720` bytes.
No full-plane client copy or approximately 25 MiB client diagnostic allowance
remains.  A residual near 0.35--0.37 seconds is only a plausibility estimate
and is explicitly unmeasured.

## 6. Public-parity contract without execution authority

The dormant source spelling retains the full public
`Estimator.predict -> _predict_corrected` call and the same returned five
components: host, native raw20, native sigma, native-J correction, and final
public prediction.  Its unconditional guards make that call unreachable in
this package.  If and only if an external user supersedes the terminal
governance with a distinct new program, independent review would have to
require fresh official A1/B/A2 workers and byte equality for all five
components, equal nonnegative B-minus-A bills under the cap, exact operation
calls, RSS below 8 GiB, and an independently set residual law.  None of that
execution exists or is licensed here.

## 7. Freeze boundary

The frozen package may contain only this derivation, the machine-readable
preregistration, the counterfactual source, the non-executing runner scaffold,
the static verifier/result, and the source-freeze receipt.  It must contain no
release, lock, worker receipt, rank selection, row binding, bundle, coordinate
artifact, result claim, target, estimator output, graph node, or submission.
Independent read-only review is required before this static counterfactual may
be cited.
