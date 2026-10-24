# Fixed entry8 join-trace information probe

Source preparation only, 2026-09-12 PDT. This document records no numerical
execution, qualified candidate, raw-error gain, runtime repair, or release
authority. Only one target-free fixed rank35/layer8 context is selected.

## Scope and actual source boundary

Use unchanged current c2 source
`current_comp_shared_gain_slice_v1/build_v1/estimator.py`, SHA256
`c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea`.
Rank384, age4, its signed factor coordinates, packed kernels and all stored
coefficients remain unchanged. No full or partial nonlinear predictor, P/NODE
head, BD update, downstream consumer, training or parameter update runs.

Admission directly authenticates the actual positive JCELL capture and its
independent watch: both three-seal chains, all source/input/artifact bytes,
the canonical rank35 weights, and the complete sealed restart packet. The
capture already established unchanged all16/all47 controls and the original
whole typed operation sequence. These facts authenticate the packet; this
new stage does not claim to repeat all16/all47.

The original JCELL consumer supplies **code-only** typed packet restoration
and validation. Its old failed A loader and subsequent scientific analysis
grant no eligibility and are never called. A/B moment archives, labels,
truth files, credentials and network access are excluded even from hashing.

Only these original entry8 objects are restored: `p`, `lam`, `previous`,
`newborn`, `shared_old`. They retain original FP32/int64 dtypes, scalar shapes,
plain FlopscopeArray wrappers, and C/F layouts. In particular, the newborn
base remains F-contiguous. No covariance, star, mean, head state, or captured
pre-plane is an input to the rebuilt tensor. The archived pre8 response/live
index arrays are explicit fixed reader queries, copied/range-checked inside
the paid context; they are not recomputed or claimed statistically independent.
Only physical W8 is copied, from the authenticated canonical weights file.

## Original reader qualification first

The worker derives an in-memory private module by adding one unique callback
immediately before original `active_basis = Qn`. The callback only retains
borrowed Q/F references in the current owned context: no arithmetic, copy,
export, replacement, mutation or zero-billed numerical sink. Removing this
insertion reconstructs **every** original c2 source byte, including all heads.
No candidate file is materialized.

The exact original scratch-initialization statements and the exact
`pp = _shared_old384_push(...)` through `fill_diagonal(cp_pre, diagonal)`
statement block are extracted from the actual last `_predict_binary`
definition, with only reversible indentation and a helper signature/return.
The original `_shared_old384_push`, `_shared_old384_read`, dispatcher and
CP diagonal function execute in that private module. The whole rebuilt
1024x1024 `cp_pre` bytes and all1024 diagonal bytes must equal the archived
pre8 arrays before trace diagnostics begin. Copies and finite scans are paid.
Captured `cp_pre` is comparison-only and never becomes a tensor operand.

Fixed entry geometry is a=896 active input coordinates, old prefix6144,
joining cohort1984, selected old prefix R=8128, inherited columns13376,
pending newest cohort1536, native pushed total14912, response rows736,
live rows896. The diagnostic includes **all8128** original prefix factors
through this join, not merely the new joiner and never the pending newborn.

## Exact represented trace and the separate shortcut

All following work is explicitly metered FP64 arithmetic. Let X0 be the
original full `p[previous,:8128]` cast to FP64, lambda the corresponding actual
stored coefficients, and Q,F the actual stored post-join FP32 arrays cast
to FP64. Form the dense diagnostic reconstruction X1=QF. Define

```
t0 = X0 [lambda * column_squared_norms(X0)]
t1 = X1 [lambda * column_squared_norms(X1)]
Delta = t0 - t1
d = Delta / (a + 2)
H_ijk = d_i I_jk + d_j I_ik + d_k I_ij
```

I is identity on the actual896-coordinate active support. Thus the partial
trace of H is `(a+2)d=Delta`, including repeated indices. This is a diagnostic
identity-star correction only; it is not installed in any physical carrier.
“Exact” here means the explicit dense FP64 diagnostic of the stored Q/F
representation, subject to reported FP64 arithmetic, not an assertion that
native FP32 reassociation equals dense FP64 evaluation.

The **approximate orthogonal shortcut** is separately evaluated:

```
Delta_short = t0 - Q [F (lambda * column_squared_norms(F))]
```

It is never called exact. Report
`||Delta_short-Delta|| / ||Delta||`, Q's measured Frobenius orthogonality
defect e=||Q.T Q-I||F, and the algebraic upper bound
`sqrt(1+e)*e*sum_j |lambda_j| ||F_j||^3` on the trace discrepancy
(with FP64 evaluation/roundoff explicitly distinguished). Zero/near-zero
trace directions cannot pass this comparison by division tricks.

## Actual W8 query proxy and fixed gate

Let w_i be the actual active-coordinate W8 column, Zs=W8.T Xs. The original
before-join tensor is the source reference, not a true population moment:

```
k_s(i)   = sum_j lambda_j Zs(i,j)^3
R_s(i,l) = sum_j lambda_j Zs(i,j)^2 Zs(l,j)
u = W8.T d;  g_i = ||w_i||^2;  D_il = w_i.T w_l
h_k(i)   = 3 u_i g_i
h_R(i,l) = g_i u_l + 2 D_il u_i
```

Repeated row/column entries of each plane explicitly use its corresponding
cubic diagonal. Compare the tensor represented by X1, and that tensor plus
H, against the tensor represented by X0: uncorrected and corrected diagonal
MSE and repeated-plane Frobenius error squared.

The primary diagonal queries are all640 W8 columns384..1023, regardless of
the reader masks. The primary plane is restricted to archived response rows
and live columns whose original indices are both >=384. These are
**sketch-unused query columns**: columns0..383 constructed the join sketch.
Selectors depend on all columns, so neither post-selection IID Gaussian
sampling, independent held-out networks, nor confidence bounds are claimed.
The full archived response/live rectangle and all1024 diagonal queries are
secondary reports, never alternate acceptance gates.

One prospective scientific gate, all conditions required:

- Both primary error denominators exceed their explicit arithmetic floors.
- Corrected/uncorrected diagonal MSE <=0.75.
- Corrected/uncorrected plane Frobenius error squared <=0.75.
- The exact trace direction is resolved above its arithmetic floor and the
  shortcut discrepancy is <=5% of that exact trace direction.

There is no direction fitting, multiplier, coefficient, rank/age/layer/query
sweep, raw-error gate or secondary-statistic rescue. A failed gate retires
this fixed information test; it does not trigger another stage.

### Fixed roundoff floors

Set `gamma=64*2^-52*(2a+2r+4R+64)` prospectively. Let
`q=sqrt(1+||Q.T Q-I||F)` and
`S=sum_j |lambda_j| (||X0_j||^3 + q^3 ||F_j||^3)`.
These absolute-coefficient Cauchy scales account for signed cancellation.
For each queried diagonal use scale `g_i^(3/2) S`; for each plane cell use
`g_i sqrt(g_l) S`. The denominator floor is the sum of squares of gamma
times the relevant scale; the trace-direction floor is `(gamma S)^2`.
All scales, reductions and comparisons are paid FNP operations. These are
fixed conservative arithmetic-resolution guards, not statistical uncertainty
estimates or a theorem about the endpoint. Unresolved denominators produce
`roundoff_unresolved` with operational completion, not artificial zero errors.

## Complete cost, memory and ownership

`cost_certificate()` computes a fail-closed, metadata-only source upper bound
before numerical import. The fixed shapes of every native join/core/push/read
product and every FP64 diagnostic product are listed in the result. Original
native products use the qualified JCELL dispatcher bound
`2mkq+96(mk+kq+mq)` with explicit >1024 row, >8192 output/contracted tiling and
concatenation/add costs.96 dominates the current D6 addition/copy coefficients;
the QR allowance is8*a*r*r. FP64 dense products cost at most4mkq.

The largest new products are dense QF, two1024x896 by896x8128 query products,
two736x8128 by8128x896 reader products, query Gram and Q Gram. The certificate
also counts every pointwise large-bank pass, casts, gathers, norms, cubes,
signed contractions, scalar checks, entry/weight/control copies and exports.
No diagnostic subtotal is subtracted from the actual bill.3BF covers fresh
original scratch/view setup conservatively; no inherited startup credit is
transferred. The admission certificate must itself be <=200BF, and the actual
one-context bill must be <=the certificate and <=200BF. This is a research
context, not a complete-estimator cost or wall-time forecast.

Explicit-array lifetime bound is3.5GB: the351MB byte packet,1.169GB existing
arena, restored entry<90MB, one native full pushed bank<62MB, dense diagnostic
banks/transports/copies and matrix temporaries comfortably below1GB, plus
bounded weight representation/source/metadata. Query products are formed
serially; their large intermediates are not retained in readouts. This is
not a bound on process/library overhead: the independent monitor must still
observe the **entire** owned tree below8,000,000,000 bytes with zero added swap.

All numerical construction, original setup/join/read, diagnostic work and
finite/export checks occur inside one actual BudgetContext. NumPy may only
read authenticated NPY/byte representations; it performs no numerical
reduction outside the paid context. The exact public saved-bill/reset helper
and closed-operation-ledger wrapper are reused. `bill.flops` is the untouched
actual meter/ledger total, equal to `complete_research_flops`. Private module,
scratch, input, packet-restored, Q/F callback, diagnostic and context owners
are checked retired before the only progress marker. No warm cache survives.

## Launch, seals and authority

No execution by default. Worker flags:

```
--run --source-sha256 WORKER --watcher-sha256 WATCH
--protocol-sha256 PROTOCOL --watch-spec-sha256 WATCH_SPEC
```

Root alone selects the four actual reviewed SHA values and launch after
inert source/API admission. OFFPY with explicit production-OpenBLAS selection
and actual module/origin/BLAS attestation; nice>=10, assertions on, bytecode
off, all six BLAS thread settings1, no fallback/custom tariffs. Full worker
and independent supervisor deadline120s, each owned context<120s, actual
resource samples<=1s, zero added swap. No concurrent numerical worker.

Exclusive `probe_v1` contains started metadata, four `call_00` JSON records,
and linked result/completion/finalization seals. One progress event is
`event=retired,call=0,rank=35,arm=trace,complete_sha256=...` only after actual
bill/reset/release and the linked complete record. The independent watch uses
exclusive `probe_watch_v1`. Existing outputs prohibit a restart/overwrite.

Result schema is `current_comp_join_trace_information_v1`. Operational
`passed/completed` requires byte-exact current CP control, complete own cost,
finite diagnostics, source/input integrity and actual retirement. Separate
`primary.scientific_passed`/`scientific_passed` reports only the fixed proxy
gate. Full/partial model calls and parameter updates are all zero. Neither
outcome establishes final raw accuracy, J, a deployable correction, bank
acceptance, upload authority or permission for another numerical stage.
