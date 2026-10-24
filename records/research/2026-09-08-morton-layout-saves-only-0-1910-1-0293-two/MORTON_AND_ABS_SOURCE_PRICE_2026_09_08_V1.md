# Morton identity retention and alternative-basis Strassen: source prices

September8,2026 PDT. Phase2 only. No new numerical kernel, depth retry,
candidate, score or complete-bill measurement. Existing frozen work is
unchanged. These are explicit circuits, not universal lower bounds.

Current source409f uses ordinary Winograd products P0..P6 and a breadth
arena. Let A=m*k, B=k*n, C=m*n, q=7/4, G_d=sum(q**j,j=0..d-1).
The source-verified warm FP32 helper bill is

    F_current = 7**d*(m/2**d)*(n/2**d)*(2*k/2**d-1)
                +(7/4)*(A+B+C)*G_d.

This includes four input linear operations and three raw copies per side
per node, seven output linear operations, and one leaf-batched product.
It excludes outer dispatch assembly, initialization and model consumers.

## 1. A constructive in-place input layout exists

The efficiency agent supplied, and root independently checked, this common
product permutation: [P4,P0,P3,P1,P2,P5,P6]. Initialize A raw slots as
[A21,A11,A22,A12] in positions0:4, and B raw slots as
[B11,B12,B21,B22] in positions1:5. The new output operand slots are

    A: [S1,A11,A22,A12,S4,S2,S3]
    B: [T1,B11,T4,B21,B22,T2,T3].

First compute S3=A11-A21 into6; then S1=A21+A22 into0 (old A21 is dead),
S2=S1-A11 into5, S4=A12-S2 into4. On B compute T1=B12-B11 into0,
T2=B22-T1 into5, T3=B22-B12 into6, T4=T2-B21 into2 (old B12 is dead).
All three raw identity operands per side survive without a per-level copy.

Repeat across d seven-way branch axes. Pending raw subtrees are basic
slices0:4 on A and1:5 on B, not unspecified data in the remaining holes.
A's raw ordering is obtained by interleaving column/row bits and reversing
the row bit; B uses ordinary quadrant bits. A paid row-major/Morton pack
is required. At the leaves all seven-way branches are initialized, so one
batched matmul remains possible. Two C owners permit the original
seven-operation unwind into raw quadrant slots0:4, followed by paid output
conversion. This is different from existing parity-lifetime packing and
the old final-leaf streaming circuit with seven matmuls.

An explicit conservative public-array packing path uses two reshapes and
one copy per input and for the final output. Transposes/basic slices are
views; all three materializations are charged, including reshape tariffs.
It gives the SOURCE subtotal

    removed identity writes I = (q**d-1)*(A+B)
    pack/unpack price          = 3*(A+B+C)
    F_Morton                   = F_current-I+3*(A+B+C).

At d4 on the baseline's two synthetic shapes:

| Shape | Removed F | Paid packing F | Net subtotal saving F |
| --- | ---: | ---: | ---: |
|1024,832,8192|64,247,040|48,168,960|16,078,080 (0.1910% of kernel)|
|832,8192,832|114,216,960|42,971,136|71,245,824 (1.0293% of kernel)|

A conservative disjoint-owner bound, excluding ordinary operands/output
and packing temporaries, is

    4*(q**d*(A+B+2*C)+.75*q**(d-1)*C) bytes:
    1,051,939,840B push /574,461,888B read at these shapes.

Cold view binding, a bounded reusable arena implementation, actual strides,
NumPy iteration, rounding and whole-model cost remain unqualified. The
83--86% non-GEMM baseline time is mostly add/subtract, not identity copies;
it must NOT be assigned wholesale to the removable I. No runtime gain or
new experiment is admitted by these prices. The simpler interval-restricted
view diagnostic is a separate, narrower unchanged-DAG question.

## 2. General-ring alternative basis is not a free arithmetic reduction

Primary source read: Schwartz--Vaknin, *Pebbling Game and Alternative Basis
for High Performance Matrix Multiplication*, equations3.1 and AppendixB:
https://epubs.siam.org/doi/full/10.1137/22M1502719 . Their square-matrix
performance does not establish FNP performance on these rectangular shapes.
The scoped Phase2 archive search found only the different DPS48-product /
dimension47 basis correction, not an executed Schwartz--Vaknin kernel.

The basis phase changes only block22 of each input, using two additions.
Its inverse output basis changes blocks12 and21 using two subtractions.
The transformed seven-product circuit uses three input additions per side
and six output additions. Across d levels, applying the external basis
costs .5*d*(A+B+C), separately from the seven-way bilinear tree.

For a straightforward breadth implementation each transformed input has
FOUR raw operands to copy, instead of the parent's three. Three additions
plus four copies therefore costs the same input amount as four additions
plus three copies. Output reconstruction saves .25*C*G_d. At minimum this
particular implementation also copies the ordinary readonly inputs once,
cost A+B, before modifying their basis. Its subtotal is consequently

    F_ABS = leaf +[(7/4)*(A+B)+(6/4)*C]*G_d
                 +.5*d*(A+B+C)+(A+B).

Even before additional binding/packing/dispatch costs, d4 push costs
8,433,889,536F versus8,417,538,304 (+16,351,232, +.1943%); read costs
6,962,012,304F versus6,921,666,752 (+40,345,552, +.5829%).
These are root's tariff calculations for the specified FNP-style circuit,
not numbers reported by the paper. A combined in-place basis layout might
remove copies but needs a separate complete construction and price. The
paper's asymptotic addition coefficient is not itself a bill reduction here.
No alternative-basis prototype or deeper schedule is selected.

The bank stays official4.982231460183304e-9 /qualified local14
4.770235434670105e-9. Neither construction supplies raw accuracy; even the
.1 scoring floor at current local raw is2.24585e-9, so the <1e-9 goal still
requires a genuine raw-error improvement.
