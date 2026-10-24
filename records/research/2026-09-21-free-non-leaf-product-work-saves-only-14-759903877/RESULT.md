# Free overhead around the current leaf products still misses the planning cut

Granting every charged operation inside the current outermost matrix-product
routines for free, except their existing leaf matrix multiplications, saves
at most **14.759903877% on35 and13.690445234% on100**. Both are below the
fixed **16.304320666%** uniform planning cut associated with3e-9 at unchanged
raw error. The standalone fixed-leaf caching route does not advance under
this test. No kernel or candidate was built.

| Network | Original complete FLOPs | Retained leaf FLOPs inside products | All other product work granted free | Optimistic complete remainder |
| --- | ---: | ---: | ---: | ---: |
| 35 | 413448696909 | 320637683249 | 61024630243 | 352424066666 |
| 100 | 416264792412 | 327518800730 | 56988503433 | 359276288979 |

The allowance is deliberately stronger than input-transform caching. It
removes transforms of unique inputs as well as repeated inputs, output
recombination, packing, geometry binding and other non-leaf work inside
_mm. All computation outside _mm remains paid. The remainder is an
arithmetic bound for this specified deletion family, not an executable
estimator, measured new bill or official-score prediction. The two reused
development traces do not certify the cost distribution on official cases.

The saved identity traces contain590 input occurrences each:428 exact
classes,162 occurrences after the first; granting transpose equivalence
gives383 classes and207 subsequent occurrences. Empty operands are included.
This establishes input repetition even though all295 complete products are
distinct. It does not establish compatible cached transforms or their net
saving. The observed input-transform arithmetic/copy subtotals are
26,911,117,884 and25,120,389,238 FLOPs; even those include unique-input work.
The existing W5 reuse is already reflected in the recorded bills.

All590 ordered product records agree between the original cost census and
the later exact identity census. Stack charges partition to the complete
bills and independent outer-product totals exactly. Four active source AST
statements certify the retained leaf operations. Installed matmul dispatch
charges each such operation once. Seven synthetic stack records cover nested
_mm frames, direct outside products, zero charges and mismatching sites.
An independent integer subtraction audit reproduces all totals and gates;
pairwise comparison independently reproduces both operand equivalence counts.
No source, meter or estimator arithmetic was changed.

Different leaf algorithms/depths, shared leaf products, changed matrix
structure/ranks, outside-work savings, numerical approximations and combined
accuracy/cost changes are outside the result. The earlier finite-catalog
allowance cannot be added to this one: their operation schedules overlap
and differ. Smaller transform-cache gains remain possible and could matter
in a separately justified composition; this result is not a universal
obstruction or a claim that every incremental cost improvement is useless.

The source review also confirms that antithetic first-layer linear controls,
first-layer quadratic RQMC and the direct Gaussian-forecast terminal-skew
model were previously executed and failed their stated gates. Their exact
scopes and partial/fidelity failures remain unchanged; no such study was
restarted. The initial preparation script encountered checkpoint85's null
study-candidate field before producing any output. Its original source and
the schema correction are preserved before the experiment seal.

Both guarded groups retired; controller3037 is terminal. No new model calls,
arrays, targets, fits, samples, GPU work, downloads or uploads. Peak process
RSS was56852480 bytes; the sparse0.5s supervisor
samples reached1376256 bytes. Zero added
swap. About0.065 MB of new
artifacts before closeout.

Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
100/100 complete,zero failures.4e-9 is banked;3/2/~1e-9 remain unmet and the
full goal is active. The qualified covariance/readout composition remains
ready and unchanged. Automatic review rejected its external upload before
process creation because it could not verify trusted authorization for this
exact private payload and AIcrowd destination. The already requested exact
approval remains unanswered; no retry, bypass or new question occurred.
