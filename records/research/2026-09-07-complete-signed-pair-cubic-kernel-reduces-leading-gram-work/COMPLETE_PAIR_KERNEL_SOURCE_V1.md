# Complete signed-pair kernel: independent algebra only

September 7, 2026 PDT. New follow-up to the global coefficient-weighted
kernel preflight. No pair/state/target arrays or numerical test executed
for this construction. It is not the independently qualified single-cube
dictionary's native API receipt, nor a revived independent-thinning arm.

## Motivation and exact identity

Selecting only a signed pair's positive cube can discard its important
difference direction. Instead make each dictionary atom the COMPLETE
unweighted tensor T=p^3-q^3 and retain both stored physical cubes. A
dictionary coefficient c becomes stored physical weights(c,-c). This has
one degree of freedom per pair rather than two independently weighted
cubes; its approximation space is different and less expressive at a
fixed physical subset, not an equally expressive cheaper solve.

Write H=(p+q)/2, D=p-q, U=H^2+D^2/4, V=2HD and t=3H^2D+D^3/4,
with componentwise powers/products. Then

    T = 3 Sym(H,H,D) + D^3/4,
    T_iij = U_i D_j + V_i H_j,   T_iii=t_i.

Sym is normalized symmetrization. For a second pair use primes and put
x=H.H', y=H.D', z=D.H', w=D.D'. The full tensor inner product is

    K_full = 3*x^2*w + 6*x*y*z + (3/4)*(y^2+z^2)*w + w^3/16.

The repeated-plane inner product is

    K_plane = (U.U')*w + (U.V')*z + (V.U')*y + (V.V')*x.

The ordered all-distinct kernel is therefore

    K_distinct = K_full - 3*K_plane + 2*(t.t').

Root derived this from the pair's cubic polynomial and repeated planes;
the efficiency agent independently verified every coefficient/orientation.
As the inner product of explicit all-distinct tensor features it is PSD
in exact arithmetic. Signed external weights are valid in its target sum.

## Paid static opportunity, not a complete bill

Four H/D, four U/V and one t cross-Gram suffice. Nine ordinary FP32
products cost

    9*(2*a*K_pair*M_pair-K_pair*M_pair).

At K_physical=2*K_pair and M_physical=2*M_pair, this leading bill is
exactly25% below the three physical-atom cross-Grams. At the same physical
retained width1024, the solve dimension is512 rather than1024. Future
physical factor push/read widths and actual old/new plane restoration
do not become cheaper merely by changing the coefficient parameterization.

The25% is NOT whole-estimator savings. H/D/U/V/t acquisition and casts,
kernel polynomial/RHS, target-independent pair selection, scaling, solve
conditioning, original-factor packing and full response-reader behavior
remain additional. No complete current-response bill was measured.

Derived FP32 H,D,U,V,t generally do not give the exact kernel of the
original stored FP32 p,q. Keep p,q unchanged and qualify against THEIR
explicit tensor, rather than an ideal reconstructed pair. A FP64 final
polynomial cannot recover information already lost in FP32 features or
dot products. No ridge or fallback policy is selected here.

## Admission status

This removes one known defect in positive-only landmark selection, not
the main unknown: whether512 complete pairs capture enough of the global
current post7 source, stably and cheaply, for actual raw times the full
scoring multiplier to improve. There is no current-spectrum, capture or
endpoint-value receipt. The old thinning results neither establish nor
rule out global coefficient-weighted compression.

No full-state/target read or real-network run is admitted solely by this
identity. A later proposal must fix selection/numerics and its complete
economics before data, keeping proxy tests distinct from actual paired
raw/full-cost evidence. The tested single-cube tiny certificate and the
response-reader caveat are preserved, unchanged. No candidate changed.
