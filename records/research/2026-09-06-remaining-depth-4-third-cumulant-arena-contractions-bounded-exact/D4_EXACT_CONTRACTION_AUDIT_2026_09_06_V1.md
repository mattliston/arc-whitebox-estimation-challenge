# Remaining D4 contractions: bounded exact-algebra audit

September 6, 2026 PDT, source-only. Current candidate:
`candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py`, SHA256
`3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757`.
No model execution, new numerical certificate, target read, fit, or edit to
any existing file. The final definitions in this append-built candidate,
not shadowed earlier definitions, were inspected.

## Finding

**No credible new exact whole-card reduction toward the approximately
19.7% needed for an unchanged-raw sub-4e-9 bank was established.** This is
not a lower bound over all algorithms. The plausible local rearrangements
either preserve the dominant bilinear work, concern too little work, or
lose a separately required intermediate.

## Surviving dominant work and ownership

Let n=1024, a=previous alive count, c=current alive count, R=total carried
columns, and s=the youngest cohort's pair count. Counts are multiples of
32. Products below describe conventional leading arithmetic only; actual
bank metering uses its D4 rectangles, shape-selected square schedules,
8192-column tiles, additions/copies, and rounding.

- **CP push:** n-by-a times a-by-R. The youngest pair's common base is
  already pushed once (n-by-a times a-by-s), with its two factors restored
  from that common result and selected rows of W. Older columns are dense
  and both signed atoms remain necessary. This is lines1277–1286, called
  by the effective predictor at lines1433/1452.
- **Repeated CP plane:** `(Q*Q) @ (Q*lambda).T`, Q c-by-R; conventional
  leading work 2*c*c*R. K21(i,j)=T(i,i,j), not T(i,j,j), so it is generally
  nonsymmetric even though the underlying third tensor is symmetric.
- **Scalar CP read:** all n outputs retain an FP64 weighted cubic sum.
  The final layer already stops here: it constructs no full pre-plane,
  nonlinear covariance, star birth, or outgoing factor state.
- **Covariance:** a-by-a times a-by-n, followed by n-by-a times a-by-n,
  plus cold diagonal transport. The intermediate is Cpost*W.
- **Star:** n-by-a times a-by-a, followed by two distinct c-by-a times
  a-by-c products, plus scalar/cold terms. Its first intermediate is
  W.T*A. It is not the covariance intermediate or the newborn common
  push: A, Cpost, and the newborn base are different matrices.

The existing current-bank `bank_matmul_census_v1.json` has this exact source
SHA and byte-identical output qualifications. The older detailed block
census `compact_gaussfirst_cost_census_r35_v2.json` is a different precursor
and must not be silently quoted as this bank's exact block bill.

## Identities examined, and why they do not establish the requested saving

### 1. Signed pairs do not halve plane width

For each pair define h=(p+q)/2 and d=(p-q)/2. With signed coefficient a,

    a*(p_i^2*p_j - q_i^2*q_j)
      = 2*a*((h_i^2+d_i^2)*d_j + 2*h_i*d_i*h_j).

Across N pairs this is two c-by-N times N-by-c products, exactly the
original 2N contraction width. Both h and d require transport after the
already-shared first step. It also changes FP32 reassociation; real-algebra
equality is not byte parity. This is already identified in
`PAIR_QUERY_ARCHIVE_AUDIT_2026_09_06_V1.md`, not a new lead.

### 2. Shared operands save packing, not the dense products

Concatenating the common-left pushes can share some left Winograd
transforms but does not reduce their m*k*n term. In the effective
`_packed_product_cached`, four left add/subtract operations and three
left copies are performed per recursive node; leaves still multiply every
right-hand side. Current views and bulk copies are already cached.
Sharing additional left transforms across 8192-wide tiles could remove
small repeated packing work, but cannot erase the dominant CP products.
It additionally requires an arena-lifetime redesign, not a free reuse of
views that are overwritten by the present right/output scratch schedule.
No unsupported wall-time or whole-bank saving is claimed.

### 3. Newborn-plane/star fusion loses a needed intermediate

At the first transport of a new pair, write its outputs as B+t*V and B+s*V,
where B is the already-shared common push and V=W[selected,:].T. Expanding
its plane creates terms with right factor V.T, so some could be combined
with the star's `(pushed*W.T) @ W` contribution if only their **sum** were
needed. That observation is not a deployable eliminated product here.

The effective predictor separately consumes `cp_pre` in
`inherited = g[:,None]^2*g[None,:]*cp_pre`, then uses this inherited plane
to define `innovation`, the next signed pair, `cp_post`, and the repaired
coordinate star (lines1484–1511). It therefore needs CP and star pieces
separately, not only their summed K21. Recovering the individual piece
requires the product the proposed fusion removed. Replacing this ownership
would change the represented all-distinct third state, not just implement
the unchanged bank more cheaply. No missing-intermediate-free schedule was
found.

### 4. Symmetry and duplicate diagonal reads are too small or inexact

Only true symmetric products such as the initial Gram, and the ideal
real-arithmetic covariance congruence, admit upper-triangle savings.
They do not make the CP plane or directed star plane symmetric. Actual
FP32 covariance transport is not explicitly symmetrized, so dropping one
triangle would additionally change numerical state. Merely skipping the
CP plane's overwritten diagonal saves only O(c*R) rather than O(c*c*R)
arithmetic, while its replacement scalar sums intentionally use FP64.
The previous fused scalar-diagonal attempt already has a preserved memory
failure in `STATUS_2026_09_04_2144_PDT.md`.

## Archive exclusions and decision

Read/rechecked `PAIR_QUERY_ARCHIVE_AUDIT_2026_09_06_V1.md`,
`DIFFERENCE_LEG_CUTS_V1_CLOSED.md`,
`SYMMETRIC_TRIANGLE_QUERY_COST_V1_CLOSED.md`,
`SMIRNOV_BANK_AUDIT_V1.md`, `RECTANGULAR_CATALOG_V1_CLOSED.md`,
`SMIRNOV_GLOBAL_BATCH_REPRESENTATIVE_V1_CLOSED.md`, and the 10:43 PDT
continuation. No D6 batching, Smirnov, active-row, thinning, rank-pruning,
or known fused-diagonal rerun was performed or recommended.

This bounded audit yields no new execution arm. Preserve the bank and
continue research only when a distinct proposal has a complete consumer
and cost accounting, rather than promoting an intermediate-only identity
as a whole-card reduction.
