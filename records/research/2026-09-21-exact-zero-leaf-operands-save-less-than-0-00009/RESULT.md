# Exact leaf zeros add negligible savings to the fixed-schedule bound

The two unchanged native runs contain **no all-zero operand batches** among
17,475,569 leaf batches. Removing every scalar product with an exact-zero
operand would save at most **350,566 / 357,346 FLOPs** on networks 35 / 100
within this fixed leaf schedule. Each is less than 0.00009% of the complete
bank bill. A substantial sparsity opportunity was not hidden in these leaves.

Even after also granting all non-leaf work inside the current outer matrix
products for free, the optimistic total savings remain **14.759988667% /
13.690531080%**, below the fixed **16.304320666%** uniform planning cut for
3e-9. Both cases fail. No sparse kernel, estimator, package, or upload follows.

| Network | Complete bank FLOPs | Leaf scalar-product terms | Terms with an exact-zero operand | Optimistic complete remainder |
| --- | ---: | ---: | ---: | ---: |
| 35 | 413448696909 | 164105764518 | 175283 | 352423716100 |
| 100 | 416264792412 | 167252818330 | 178673 | 359275931633 |

For each classical leaf batch, T is the exact number of nonzero scalar
products, obtained as the sum over contracted coordinates of the two
operand support counts. If Q outputs have any such term, ordinary sparse
accumulation costs T multiplies plus T-Q additions. Granting the maximum
Q=min(T,m*n) gives the lower count T+max(0,T-m*n). This is an optimistic
bound for zero-term deletion in the existing term circuit. It does not
assert that sparse dispatch, indexing, copying or complete inference could
achieve that count. All outside-matrix-product work remains paid.

All 795 charged leaf calls match the installed dense FP32 tariff and all
four original source-site totals exactly. Every one of the 590 outer call
records matches the previous shape/dtype/caller/bill census. Both predictions
match every one of the 16 original output rows and their complete cold bills.
Both callbacks were restored and native numeric caches retired.

Streaming count fixtures include signed zeros, sparse/disjoint supports,
batching, strided/transposed inputs, permutation, a buffer larger than 1MiB,
exact repeats, input-byte preservation, and malformed/nonfinite rejection.
The real FNP fixture preserves its output and bill. Independent exhaustive
scalar enumeration covers 4,420 binary matrix pairs and verifies the term
factorization and lower-bound inequality. The metadata audit reproduces
the complete integer accounting and gates. A separate post-result aggregate
replay derives 2T-min(T,batch*m*n) and exactly matches the recorded sum of
per-batch lower counts for every one of the 795 calls; no additional arrays,
targets, parameter choices or changed gates were involved.

This result excludes neither different leaf algorithms/schedules, reuse of
nonzero products, special-constant operations, cancellation of nonzero terms,
other structured representations, changed physical states, nor smaller
useful cost/accuracy compositions. These are two reused development traces,
not a distribution-wide bound or an official-score forecast. The earlier
finite-catalog and free-overhead allowances overlap and must not be added.
The source review also explains why arbitrary child schedule choice cannot
improve an unrestricted additive-cost optimum when every child has the same
shape; structural zeros and shared products are outside that observation.

Both full native calls used 829,713,489,321 measured
FLOPs. All five guarded groups retired; controller 11458 is terminal. Peak
sampled group RSS was 6,457,212,928 bytes,
peak process RSS 6,457,606,144 bytes, with zero added
swap. About 0.964 MB of new
artifacts was preserved before closeout. No truth labels, fitting, new
sampling, GPU work, downloads or uploads were used.

Official 331684 remains adjusted 3.5844144212297072e-9 and raw
1.949715441540434e-8, 100/100 cases, zero failures. The 4e-9 milestone is
banked; 3e-9, 2e-9 and approximately 1e-9 remain unmet. The full Phase 2
goal remains active. At unchanged raw error, the 0.1 multiplier floor rules
out approximately 1e-9 from cost alone, but does not rule out 2e-9.

The separate covariance/readout release is unchanged. Automatic approval
review previously rejected its AIcrowd upload because it could not verify
trusted authorization for that exact private payload and destination. The
existing exact-upload approval question remains unanswered; there was no
retry, bypass or new upload attempt in this study.
