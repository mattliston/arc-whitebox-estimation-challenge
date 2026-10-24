# Last-two-layer dependency pruning: exact small dead work, no large saving

September 6, 2026 PDT. Phase 2 source/algebra audit only. Layers are zero
based: the retained prefix ends at post13; the remaining nonlinear layers
are14 and15. No network/target/payload array decoding, estimator execution,
rewrite, refit, numerical test, or reopening of a closed arm.

## Verdict

The existing pruning is NOT literally exhaustive: newest-factor replicas,
one diagonal-only temporary, and two overwritten/unused calculations can
be removed. Their conservative last-two-body bill envelope is below0.04%
of the archived current-bank total. None removes a GEMM. No exact large
contraction saving toward a sub4e-9 bank was established.

All dominant products at pre14 and pre15 have live consumers. Terminal
streaming can reduce intermediate storage, but does not eliminate their
arithmetic and has unqualified floating-point reduction-order effects.
This is a circuit audit, not a universal lower bound on tensor algorithms
or a measured wall-time/memory result.

## Sources and existing exclusions

Current candidate: `candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py`,
SHA256 `3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757`.
Inspect the last `_predict_binary` at1409, shared push at1277, actual
`_nonlinear_pair_batch` at643, scalar map at704, and P/Q wrapper at778.
Earlier shadowed predictor definitions are not the effective circuit.

Read `D4_EXACT_CONTRACTION_AUDIT_2026_09_06_V1.md`,
`CP_SUFFIX_RANGE_PRODUCT_SOURCE_AUDIT_V1.md`,
`PAIR_QUERY_ARCHIVE_AUDIT_2026_09_06_V1.md`, the shared-newborn builder,
and the current scalar-final/head source. This does not reopen global
cap/age/rank schedules, response-row approximation, triangle queries,
Smirnov, D6, or a known final-diagonal variant.

The existing current-source `bank_matmul_census_v1.json` is target-free
shape metadata, SHA256
`22479013b467f8e8c3fa8f4322915da5c2a9c352f9bbf78f1f4fc8823876e3a5`.
Its archived full bills are490,466,018,197 on rank35 and489,374,517,341 on
rank100, with original output hash parity. Shapes are aggregated across
the whole run, NOT per-layer receipts. No newly measured terminal subtotal
or timing is inferred from the ordering of those dictionary keys.

## Exact consumer ownership

| State/read | Immediate use | Why terminal information remains live |
|---|---|---|
| Post13 mean/C | W14 mean and full covariance congruence, including cold diagonal | Pre14 Gaussian gate, pair correlations, marginal moments, and head14 features |
| Post13 CP bank and newborn13 | Full n-row push, active repeated plane, all scalar cubics | Pre14 nonlinear map and head14, plus inherited factors for final cubic queries |
| Post13 star and trace | Active pre14 repeated plane plus all diagonals | Same pre14 moment/head reads; not merely final diagonal consumers |
| Pre14 full C and K21 | Gaussian6/cubic2/fourth pair map and30 features | P0 head14 uses normalized K21 row/column sums and squared norms, correlation means/absolute/squared statistics, and rho*alpha |
| Post14 map covariance before head | Radial reset and clipped post-correlation features | These features determine the new state14 and mean correction |
| Head14 mean correction/state | Corrected mean14, consistent C14 update, final P-message inputs | Both delta14 and all retained state14 channels have live final paths |
| Corrected/masked C14 | Final diagonal covariance query and active trace metric | `v15` and `3*bv15*(W15.T*b14)` |
| Post14 repeated-third target | Birth diagonal bookkeeping and repaired coordinate star | Final star cubic query uses its offdiagonal entries through W15 powers |
| Post14 radial scalar | Final k4=`radial14*v15^2` | Final host and Q features, including post-v/k3/k4, all use it |
| Post14 inherited CP/newborn14 | Final n-output cubic queries | Final k3 and Q features; no terminal plane is computed |

At15 the bank already skips a full covariance congruence, active repeated
plane, nonlinear pair map, star birth, and outgoing factors. It computes
only mu15/v15/k3/k4, the scalar map, and final Q. The scalar map's r2/r3/r4
and post-v/k3/k4 are NOT dead: final_scalar_features includes those fields.
The final feature mask does not suppress the full pair-statistic features
at14; they were consumed earlier to construct state14.

## Why small downstream contractions do not remove upstream pair work

Let a13/a14 be the respective alive counts, b13/b14 birth counts, and R13
the factor count arriving at pre13. Post13's inherited prefix has R13
columns; shared newborn13 adds2*b13 at pre14. With R14=R13+2*b13:

- Pre14 CP transport has a full n-by-a13 times a13-by-R13 old-factor
  product and one n-by-a13 times a13-by-b13 common newborn product.
- Pre14 plane has a14-by-R14 times R14-by-a14 work. Head14's squared
  normalized plane statistics prevent replacing this by only W15 queries.
  A Gram in the factor index to obtain row norms instead introduces an
  R14-by-R14 object; no cheaper exact alternative was established.
- Pre14 covariance requires both products of its active congruence and
  the cold diagonal contribution. Full Cpre14 enters entrywise rho powers
  and head statistics. Covariance/star/newborn have different left state
  matrices; no shared identical intermediate eliminates these products.
- Pre15 inherited CP transport is n-by-a14 times a14-by-R14, plus the
  newborn14 common product. Every final output has a cubic sum; there is
  no source-proven lower-dimensional final query set to substitute.
- Pre15 covariance and coordinate star each already use one matrix
  product followed by a diagonal contraction, with cold scalar terms.

Post14 target/C can be streamed into those final contractions instead of
stored densely, but producing the target requires the same pair-map
entries. Multiplying each separate Mehler term by W15 before summing
duplicates its matrix contraction; it does not remove the one product
currently performed after their sum. The head additionally needs clipped
post-correlation statistics. Keeping only diag(C14) is not an exact variant.

The full post22 matrix is consumed only by its trace and diagonal override,
not carried as a fourth tensor. A reduced/streamed implementation may
avoid storing it. Its pair terms still require all pair entries, and the
current three-consumer batch shares their arithmetic. No whole-bank
FLOP or wall saving for a replacement circuit is claimed here.

## Literal dead work: an exact restricted elimination circuit

These source-level eliminations apply at least at post13/post14 (some also
apply earlier). They are NOT rank pruning or changes to represented atoms.

1. `_push_bank_shared_newborn` reads only `p[previous,:old_columns]`, then
   reconstructs both newest tails from `(base,t,s)` and selected W rows.
   The separately constructed p1/p2 replicas and their appended p tails
   at1500-1507 are never read numerically at the next step. Keep the same
   newborn tuple and lambda array, but the numerical p buffer need retain
   only `g.astype(float32)[:,None]*pp`, followed by the SAME float32 alive
   multiplication. Do not combine/move those multiplications. Preserve
   the logical column count separately if diagnostic metadata requires it.
   Every subsequent `_mm` operand, shape, and operation on live columns
   is unchanged; omission only removes unused storage/materialization.
2. `residual=target-inherited-trace_post` at1490 is consumed only through
   `innovation=diag(residual).copy()`. Compute the same two subtractions
   in that order on the three diagonals. The later full `rest` is a
   different expression and remains needed for star transport.
3. The identity-metric radial calculation at693 is immediately overwritten
   at697 because the actual caller uses metric='covariance'. Remove only
   that overwritten calculation, not the covariance-metric reset.
4. The rho-power update after r=6 in `_nonlinear_pair_batch` is never read.
   Remove only that final recurrence update, not any r<=6 contribution.

Using the installed pricing convention documented in
`LEARNED_RESPONSE_PAIR_MEMORY_ARITHMETIC_CENSUS_V1.md` (FP64 rate2;
copy/cast/assembly charged; advanced gather4 before dtype rate), newest-tail
removal costs at most `4*n^2+24*n*b+n*R+O(n)` per body. For post13/post14,
`b<=n`, `R13<=26*n`, and `R14<=28*n`. The other three eliminations are
at most20*n^2+O(n) over both bodies. The combined leading envelope is
130*n^2+O(n); allowing144*n^2+16384*n gives167,772,160 charged operations
at n1024, below0.035% of either archived full-card bill. This is an upper
opportunity bound for THESE eliminations, not a new metered candidate or
a bound against all possible terminal algorithms. It is far below the
roughly20% whole-card change needed for unchanged-raw sub4e-9.

## A terminal streaming opportunity is not an eliminated contraction

At pre14, g14 and alive14 depend only on mu14 and diag(Cpre14), known
before K21/head14. Therefore each pushed CP block can both contribute to
the pre14 plane and pass through
`W15.T @ [alive14*g14*pp14_block]` for the final inherited cubic sum.
Only newborn14, trace14, and star14 need await the later head/covariance
steps. This permits earlier retirement of inherited factor buffers.

Both large pushes and the active pre14 plane remain. Moving g onto W15
instead changes FP32 rounding; summing terminal cubics per block changes
the wide FP64 reduction tree unless explicitly mirrored. Neither has a
current byte/numerical certificate. Thus this is a possible storage
schedule, not an exact measured runtime improvement or permission to
rerun closed fast-multiplication candidates.

Conclusion: no dominant contraction became dead merely because only two
nonlinear layers remain. The real literal dead work above is small and
precisely bounded. Preserve the bank and existing closeouts; no rewrite,
test, refit, or submission follows from this source-only audit.
