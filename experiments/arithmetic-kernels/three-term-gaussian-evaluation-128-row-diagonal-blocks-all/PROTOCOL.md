# One fixed Gaussian-3 and batched diagonal-128 child

Parent is immutable batched Gaussian-4 candidate e82f7bf1. Its failed boundary
receipt remains failed. This study makes exactly three numeric source edits:
Gaussian pair slices 1:5 to 1:4, DIV6 length 4 to 3, and X1 scratch view
(4,3,n,n) to (3,3,n,n). The Gaussian comment is updated separately. PB7,
FS/GS derivative orders 0 through 6, RR, P12, E2, EL2, X2/X3/T/U and learned
coefficients are unchanged. No order, amplitude, depth or partition search is
authorized. Build metadata records every replacement and its exact inverse.

Append only the fixed P4 diagonal groups [128,128] instead of [128,64,64],
and add output-side key C(128,128). The three outer rectangles, D6 kernel,
original 391602176-float arena, 649 old single-side keys and eight old batch
keys are unchanged. Expected setup is 682 bound entries: 24 KS, 649 fused
single sides and nine fused batch sides. Cache caps and reset semantics do
not change. The new key is a view into the old C region; no extra tree arena
or packed inputs are allocated. Output buffers remain external, and every
upper/mirrored block is copied before the next arena writer.

The helper is 717329488 paid warm FLOPs, a source forecast of 444 operations
versus the measured parent's 709895628 FLOPs and 537 operations. There are
14 expected helpers, so the geometry costs 104074040 additional FLOPs and
removes 1302 source-level operations. This is not an RPC or residual claim.
The one new setup side costs 1997072 FLOPs. Gaussian direct fixed-shape
savings are 161480704 FLOPs over 14 calls, plus 3145728 if the X1 view is
bound once. The one-element arange difference and conditional downstream
routes are not included. Actual full native bills are authoritative.

Root alone executes in order: primitive, native2, boundary2, fixed native6,
then broad40 only after each frozen admission passes. The primitive retains
the previous integer and Gaussian covariance fixtures and seeds. It requires
integer output exactly equal to the dense reference, Gaussian relative
Frobenius error <=2e-4, exact repeated child bytes, immutable inputs, exact
717329488 warm bill, setup 682/649+9, original arena identity, and empty caches
after reset. The new partition changes rounding, so cross-parent output byte
parity is not required. The primitive does not exercise Gaussian model order;
native endpoint scoring does. Its guard is 45 seconds, RSS <7.5GB, zero added
swap and complete retirement, with all BLAS thread variables fixed at one.

Source preparation authorizes no execution. Root's prospective screen and
admission artifacts separately fix score gates and job commands; the source
must not change after those outcomes. Final release requires a fully validated
gain >=1%, with unchanged local resource gates: setup <=4.1 seconds, residual
<=.35 seconds, projected whole wall <=105 seconds, RSS <7.5GB, zero added swap
and <70000 handles. No failed predecessor gate is relaxed or overwritten.
