# E1 GN channel batching, prospective source preparation

Parent is frozen E1 `6ac194efda6491b48f86eca074493c503f9312c6fd9e667105f378ee7637d6f9`.
It passed native2 versus 88683 but failed early boundary residual projection
0.357050751 > 0.35. That failure remains false, with no unchanged native6/8.
This is one distinct dispatch refactor, not a revised threshold or scientific
variant. Gaussian3, plane E1, EL2, all learned literals and routes stay fixed.

Only the final active GN implementation is changed, plus an appended cached-view
helper/reset wrapper. The existing identity-checked fallback is unchanged.
The two channels are evaluated in four stages: base outer products, tail outer
products, rho times tail, then base plus tail. The two completed channels are
multiplied by their original dT values, divided by FP32 [2,6], then channel0 is
explicitly added to channel1. No sum reduction, FMA, reassociation or reciprocal
replacement is introduced. All scalar/vector formulas and the final
`(X+X.T)*ss32`, cast to F64, diagonal zero are unchanged.

dT21 writes to a scratch plane with ordinary divide(out=). dT31 uses
`multiply(weight, rho.T, out=D1.T)` with cached transpose views: every entry
remains weight_i*rho_ij. This avoids the installed einsum(out=) materializing
copy. Exact backend bytes, including signed zero, must pass the primitive;
the source argument alone is not a byte-parity certificate.

Scratch: D, base and tail each occupy two planes in [0,6*n*n), inside the
existing pair-map idle [0,9*n*n) prefix. Pair returns before GN; pair output
covariance/marginals are separate allocations and T/U are no longer consumed.
GN calls no packed kernel. FT/PB are separate persistent buffers. Cached basic
slices retain actual arena/FT/PB identities; mismatch rebinds, and parent reset
clears the new cache. No fancy-index copies, new arena or setup kernel/side key.
The original setup counts remain 682 bound,649 lvfuse,9 psdb_fused.

Warm target: thirteen arithmetic calls become seven; the unused FT basic slice
and unstack disappear. Across thirteen GN calls,104 explicit FNP calls are
removed. Arithmetic remains13*n*n F32 FLOPs per contraction; three new scratch
reshapes bind6*n*n elements once, plus tiny divisor construction and free basic
views. This is a dispatch/temporary-handle proposal, not a timing guarantee.
The prior residual deficit corresponds to1.7627ms at the binding local row;
new boundary measurement decides whether this refactor is useful.

Root-only primitive: `NATIVE -B probe.py BUILD_SHA RESULT`,60s/1GB/zeroSwap,
BLAS1. It extracts actual helpers/statements, tests zero/signed-zero/dyadic/
deterministic random finite F32 inputs, exact parent/child outputs, fresh
changed FT/PB contents, direct factor writes, actual arithmetic operation/FLOP
counts, borrowed view addresses, untouched neighboring scratch, and reset.
 No estimator import. Native acquisition requires all16 outputs byte-exact
against failed E1 parent on existing ranks35,100. Across fixed eight, require
L0 byte parity against88683. No unchanged E1 six-network run is authorized or
needed solely for parity; primitive/source proof plus two full endpoint
comparisons qualify the refactor, not a claim of measured eight-network parity.

Prospective score comparator is frozen88683. Native2 requires strict adjusted
improvement and both wins; then exact-source early boundary must pass before
fixed native6/8. Native8 component requires >=0.05% adjusted gain,>=6/8 wins,
CI upper<0. Existing engine50,000 paired bootstrap seed20260926. No raw veto.
Native2/6 guards150/300s,7.5GB,zeroSwap; setup<=4.1s; early residual<=.35s,
projected wall<=105s and CPU agreement<=5percentage points. Final bank remains
full40>=1% versus latest official,>=28wins,CI<0,max regression<=10%,plus serving.
No new order/scope/coefficient variants, retries, grid or gate relaxation.
Source preparation does not itself authorize any execution or release.

## V2 harness-only repair

V1's primitive reached exact first-case output/factor parity, then failed its
operation-count assertion because the installed tariff labels division
`true_divide`. V1 source, failed primitive receipt/log/guard remain preserved.
V2 adds only `true_divide` to the primitive arithmetic-name set; no numerical
criterion or candidate arithmetic changes. The candidate must remain byte-exact
SHA06538586dd306df2b00986f43cc85afd39d515a12c8977f4da6fa9c539ed91b1.
This new study owns fresh build/probe/guard/receipt paths; it does not retry or
overwrite the failed V1 canonical receipt. All prospective gates above remain.
