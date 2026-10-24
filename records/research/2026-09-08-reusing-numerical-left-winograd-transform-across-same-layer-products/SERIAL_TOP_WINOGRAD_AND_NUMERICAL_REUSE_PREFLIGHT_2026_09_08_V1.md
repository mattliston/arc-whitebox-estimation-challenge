# Serial top-child Winograd and numerical-transform reuse: price preflight

September8,2026 PDT. Phase2 only. Existing efficiency agent supplied an
independent source/lifetime/count review; root read current1664--1769 and
packed_arena_generic.py and checked the arithmetic structure. No kernel,
array, estimator, benchmark or changed candidate was executed. Old D5/D6,
Smirnov and other closeouts are not reopened by this note.

Current qualified source is unused_materializations_v1/estimator.py,
SHA409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f.
Local14 adjusted4.770235434670105e-9, mean466058731667.9286F; official
329926 remains4.982231460183304e-9. No new measured score is claimed.

## 1. Serial top children are genuinely different from lifetime packing

The current cached helper is byte-identical to the corresponding D5/
bulkcopes helper: all7^level descendants are transformed, ONE batched
leaf matmul runs, then every descendant is unwound. Existing lifetime
packing aliases dead storage; it does not execute top children serially.
No serial-top execution was found in the inspected lineage. This is a
bounded source-absence statement, not an exhaustive archive claim.

A safe new circuit can construct all seven top operand pairs using the
same bulk-copy primitives, execute each with a breadth-batched depth(d-1)
child kernel sharing one child arena, write directly into disjoint top
product slots and apply the original top reconstruction. It need not add
product copies. This retains the same arithmetic DAG in real arithmetic;
FP32/library/consumer equivalence is NOT newly qualified.

Let A=m*k, B=k*n, C=m*n, q=7/4, with all dimensions2^d-aligned. The
unchanged FP32 kernel arithmetic bill is

    F_d = 7^d*(m/2^d)*(n/2^d)*(2*k/2^d-1)
          +(7/4)*(A+B+C)*sum_{j=0}^{d-1} q^j.

This includes this helper's counted arithmetic/copy elements, not outer
dispatch assembly, padding, setup or a complete estimator. Each forward
level has six left and five right FNP primitives; each unwind has seven.

| Depth | Current warm kernel calls | Serial-top calls | Leaf calls |
| --- | ---: | ---: | ---: |
|4|73|403|1 to7|
|5|91|529|1 to7|
|6|109|655|1 to7|

The formulas are18d+1 and18+7*(18*(d-1)+1). Metadata/view creation,
cache misses, releases and initialization are additional. Cached stable
views do not eliminate these numerical primitives.

For the ARCHIVED representative shape(1024,832,8192), not a newly
captured409f trajectory, D5 replaces one batch of16807 leaves with seven
batches of2401. Each leaf remains32-by26 times26-by256. Kernel arithmetic
is7,599,277,504F versus D4's8,417,538,304F. Transposition changes orientation,
not this price. D6 needs separately paid64-lattice handling where the
current supports are only32-aligned.

### Scratch is smaller, logical traffic is not

The existing arena size in FP32 elements is

    S_d=(q^d+q^(d-1))*(A+B)+q^(d-1)*max(A,B).

A safe disjoint serial-top layout has

    H_d=q*(A+B)+q*C+(3/4)*C+(1/4)*S_(d-1)(m,k,n).

These own top operands, top products, reconstruction scratch and one
child arena. The ordinary output stays outside, matching the parent
counting convention. At the maximum read tile1024-by8192 times8192-by1024:

| Schedule | Scratch bytes |
| --- | ---: |
|Breadth D4|1,168,900,096|
|Breadth D5|2,045,575,168|
|Serial-top D5|420,151,296|
|Serial-top D6|639,320,064|

These are scratch-owner counts, NOT whole-process RSS predictions.
The non-GEMM logical reads+writes remain

    4*((9/2)*(A+B)+(21/4)*C)*sum_{j=0}^{d-1}q^j bytes.

For the representative push, that is3,509,975,040B atD4 and
6,456,635,904B atD5. Serializing top children does not change this total.
Physical cache/DRAM traffic, GEMM-internal traffic and throughput are
not established by counting logical accesses.

D5 WITH bulkcopies already had one official120s total-wall timeout in
329884, not a memory/residual failure. The serial circuit reduces peak
ownership but adds calls and establishes no timing improvement. It earns
no deeper-kernel run on novelty alone. Matching timing breakdowns are a
separate read-only archive question; any subsequent finding should be
an additive note, not a silent change to this preflight.

## 2. Reuse actual left transforms, not merely scratch views

Within a layer, L=W32[previous,:].T is genuinely identical across the
active covariance outer product, star push, old-CP output tiles and
shared-newborn push (current333,346,1289--1294). Their RHSs are different;
no dominant leaf product is shared. The current frame cache stores views,
not numerical transforms. Old common-left two-channel code avoided a raw
input stack but still ran both transforms; its20,971,520F saving was NOT
cached numerical left leaves.

The opportunity is already acknowledged in D4_EXACT_CONTRACTION and
CURRENT_RESPONSE_EXACT_REUSE; it must not be called a newly discovered
large saving. A paid implementation would preserve the first product's
left leaf in a separate same-stride FP32 owner before the arena unwind
overwrites it, and reuse it only for the same L/depth/layout in that
layer/net. No unmetered cross-net numerical cache is proposed.

For t same-depth uses and gamma_d=sum_{j=1}^d(7/4)^j, charging one copied
left leaf gives the incremental saving

    ((t-1)*gamma_d-(7/4)^d)*n*a.

Here n-by-a is L. A shaped fnp.empty has no data-initialization charge;
the copy is paid. No reshape is needed in this construction; any added
reshape/materialization would need its own charge. The maximum leaf owner
is39,337,984B atD4 or68,841,472B atD5 for full-width L. The cache must
outlive the current arena overwrite, and its full-process memory effect
is unqualified.

A generous static bound for this SPECIFIED set of same-L opportunities:
CP pushes<=47, covariance outer15, star15, at least15 creators, so at most
62*gamma5*n^2=2,338,072,576F gross. Adding resident-Gram top-block duplicates
<=301,686,784F and duplicate W gathers<=31,457,280F gives

    <=2,671,216,640F =0.5731502% of current panel mean,

BEFORE cache-copy costs. This overcredits D5 and potential same-depth
matches; it is not a complete saving or native measurement. It is not
milestone-scale. No standalone implementation/qualification is selected.

The response CP plane uses selected FP32 pp-squares and weighted pp;
the marginal cubes use full-support FP64. Those operands cannot be
silently interchanged to claim a further common product. All old parity,
precision, full-cost, memory and release obligations remain unchanged.

The two source questions supply a memory trade and a small cost ceiling,
not a forecast below4e-9. No substantial experiment is forced; the strict
official1e-9 goal remains active and unmet.
