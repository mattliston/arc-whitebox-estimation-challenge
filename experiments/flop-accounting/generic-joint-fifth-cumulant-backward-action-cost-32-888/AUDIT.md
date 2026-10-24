# P2R0F3B1B audit: output-specific contraction of a joint K5 birth

Status: **CLOSE THIS BACKWARD SPELLING; REDUCE TO A FUSED TRIVARIATE
SOURCE/ACTION EXTERNAL CONSTRAINT**  
Date: 2026-08-26  
Scope: target-free algebra, exact small-width parity, and static cost/rank
arithmetic only

## Decision

Contracting a born fifth cumulant directly against future endpoint weights can
avoid *storing* the complete K5 tensor.  It does not create a new backward
operator.  For any frozen source it is exactly the transpose of inserting that
source into the forward linearized state recursion: ordinary first-order
Duhamel.

There is nevertheless genuinely new source content beyond B1A.  The hierarchy
is:

1. The smallest joint K5 shell relative to B1A is the complete support-at-most-
   two shell `[5]+[41]+[32]`.  B1A used only `[5]`; `[41]` and `[32]` are
   algebraically independent of both its marginal K5--K12 ladder and K22.
   This shell is already implemented by the exact ledger as `k5d/K41/K32`, so
   it is not a new repository state.  Its autonomous level-D carry was unstable.
2. The smallest complete K5 shell absent from both B1A and that ledger is the
   exactly-three-distinct shell `[311]+[221]`.  It has

   ```text
   N311 = n*C(n-1,2)
   N221 = C(n,2)*(n-2)
   N3   = n*(n-1)*(n-2)
   ```

   independent symmetric entries.  It is the first honest B1B source test.
3. This three-distinct shell already closes the generic output-specific form.
   The ordinary contraction with the immediate K5-to-mean future action has
   tensor-network treewidth three and generic action rank `n`.  At width 1024
   and 15 source layers, even the favorable one-multiply/one-add ledger is
   `32,888,774,983,680` FLOPs, about 14.96 complete `2^41` budgets.  It excludes
   source generation, the full-state VJP, response functions, and the host.
4. The repository also lacks a covariance-resummed trivariate ReLU reset for
   `[311]` and `[221]`.  The exact ledger ends at bivariate `K41/K32`; the
   partition-complete audit explicitly finds no general correlated 3D--5D K5
   evaluator.  Reversing the contraction does not provide that missing source.

The graph-ready outcome is therefore:

```text
B1B backward novelty                              CLOSE
joint K5 source vs B1A/K22 novelty                PROVED
support<=2 [5,41,32]                              ALREADY LEDGERED
complete 3-distinct [311,221], generic action     OVER BUDGET
complete 3-distinct covariance-resummed birth     UNDEFINED
only reopen condition                             FUSED-K5-3D-ACTION certificate
```

No target, fit, estimator, board, submission, production run, or canonical
graph file was opened or changed.

## 1. Exact adjoint identity

Let the retained baseline state obey `s_(l+1)=F_l(s_l)`.  Add a formal born K5
source `epsilon E5 b_l` and linearize:

```text
delta s_(l+1) = J_l delta s_l + E5 b_l,
J_l = D F_l(s_l).
```

For endpoint coordinate `o`, let

```text
lambda_D,o = partial y_o / partial s_D,
lambda_l,o = J_l^T lambda_(l+1),o.
```

Then, identically,

```text
delta y_o = sum_l <lambda_(l+1),o, E5 b_l>.          (1)
```

Equation (1) is the transpose of forward Duhamel.  It can reduce memory and
allocate work by output, but it cannot make a fixed source algebraically new.
B1A's reverse/forward parity at `1.9244557334329387e-16` is the measured version
of the same identity for its reduced marginal source.

The new information considered here is `b_l`, not the direction of (1).

## 2. Collision shells and a strict non-equivalence fixture

A symmetric fifth cumulant has collision classes

```text
[5], [41], [32], [311], [221], [2111], [11111].
```

B1A consumes only marginal `g5_hat`, hence only `[5]`.  The exact ledger's
level D adds the complete bivariate shell `[41]+[32]`.  Its source fields are
`K41[i,j]=cum(x_i^4,x_j)` and `K32[i,j]=cum(x_i^3,x_j^2)`.

The first shell not already present is

```text
B311[i,{j,k}] = cum(x_i,x_i,x_i,x_j,x_k),
B221[{i,j},k] = cum(x_i,x_i,x_j,x_j,x_k),
```

with the named indices distinct.  Both partitions are required: retaining one
is not a complete exactly-three-distinct K5 source.

The exact verifier constructs a formal cumulant fixture with all diagonal,
bivariate, and order-four fields zero but one `B311` entry nonzero.  Its dense
affine/readout contraction is nonzero.  B1A and K22 both emit zero on that
fixture.  This proves source-level non-equivalence without an accuracy target.

It does not contradict (1): forward and reverse evaluation of this *new* source
remain exactly equal.

## 3. Exact affine/action contraction

For one affine map `W`, the diagonal pushed K5 consumed by a marginal mean
response is

```text
d_a = sum_(i1,...,i5) B[i1,...,i5]
      W[i1,a] ... W[i5,a].                            (2)
```

If `H[a,o]` is any frozen future linear response from `d_a` to endpoint `o`,
then

```text
delta y_o = sum_a H[a,o] d_a
            = <B, A_o>,
A_o[i1,...,i5] = sum_a H[a,o] product_q W[iq,a].      (3)
```

On canonical symmetric entries, (2) becomes

```text
[5]    B5[i]       W_i^5                              multiplicity 1
[41]   B41[i,j]    W_i^4 W_j                          multiplicity 5
[32]   B32[i,j]    W_i^3 W_j^2                        multiplicity 10
[311]  B311[i,j,k] W_i^3 W_j W_k                      multiplicity 20
[221]  B221[i,j,k] W_i^2 W_j^2 W_k                    multiplicity 30.
```

`verify_audit.py` expands a width-four dense order-five tensor, applies (2),
and compares all three endpoint coordinates with the canonical contraction in
(3) using exact rational arithmetic.  It also verifies the general finite
Duhamel transpose identity exactly.

## 4. Rank and treewidth obstruction

For a scalar endpoint, the `[311]` action tensor contains

```text
A311[i,j,k] = sum_(a=1)^r g_a U[i,a] V[j,a] V[k,a],
U[i,a] = W[i,a]^3,  V[j,a] = W[j,a].                  (4)
```

The analogous `[221]` expression changes the elementwise powers but has the
same contraction graph.  For a generic square `W` and nonzero `g`, the mode-1
unfolding of (4) has rank `n`.  The verifier exhibits an exact Vandermonde
width-four witness with rank four; the determinant witness implies that rank
loss is a proper algebraic subvariety, so generic rank is `n`.

The factor graph for contracting a generic dense source `B(i,j,k)` with (4)
has variables `{i,j,k,a}`.  `B` makes `{i,j,k}` a clique and the three action
factors join `a` to each source index.  The primal graph is `K4`, with exact
treewidth three.  Every ordinary variable-elimination order therefore forms an
`n^3` intermediate and performs theta(`r*n^3`) arithmetic.  This is a tensor-
network contraction bound for this spelling, not a universal lower bound on
all possible arithmetic circuits.

For comparison, the bivariate `[41]+[32]` contraction has primal graph `K3`,
treewidth two, and theta(`r*n^2`) work.  This is why the bivariate shell is
affordable while the first unledgered shell is not.

## 5. Width-1024 exact arithmetic

Freeze

```text
n = 1024, source layers L = 15, generic action rank r = n.
```

The canonical entry counts are

```text
N311 = N221 = 535,299,072
N3 = N311 + N221 = 1,070,598,144.
```

Counting only one multiply and one accumulation for each source-entry/action-
rank pair gives

```text
C3(r,L) = 2*L*r*N3.

C3(1024,15) = 32,888,774,983,680 FLOPs.
2^41         =  2,199,023,255,552 FLOPs.
```

Thus the favorable contraction-only count is `14.956x` the whole budget.  It
does not include powers, multiplicities, source evaluation, adjoint formation,
full-state transport, later collision sectors, or the existing host.

Using the historical no-eigh host bound `1,752,673,252,727` leaves
`446,350,002,825` FLOPs.  Under the same favorable ledger, all 15 source layers
require

```text
r <= floor(446,350,002,825 / (2*15*N3)) = 13.          (5)
```

Even spending the entire board budget and pretending every other operation is
free permits only `r<=68`.  The exact generic witness is `r=n=1024`.

Materializing the two source sectors alone takes `4,282,392,576` float32 bytes.
Source plus an equally dense action takes `8,564,785,152` bytes, leaving only
`25,149,440` bytes under 8 GiB for the whole host.  Streaming avoids this memory
failure but not the arithmetic or source-oracle failures.

The complete bivariate contraction is much smaller:

```text
N2 = 2*n*(n-1) = 2,095,104,
2*15*n*N2      = 64,361,594,880 FLOPs.
```

That number prices only the contraction.  It does not prove an integrated
full-state VJP and bivariate reset fit, and the state is already present in the
ledger rather than a new B1B representation.

## 6. Missing covariance-resummed birth

The exact ledger has a finite bivariate Mehler evaluator and emits `K41/K32`.
It has no `K311`, `K221`, `K2111`, or `K11111` field or reset.  The independent
partition-complete audit found no general correlated 3D--5D ReLU K5 evaluator.
The later independent-base IG-LC3 candidate is a different formal truncation;
its independent constraint receipt closes it as an exact/production evaluator
because it misses the Gaussian `[11111]` tree shell and fails the reached
correlation domain.

Consequently a direct contraction cannot even query the complete three-
distinct born source under the current mathematical authority.  Treating the
source as an arbitrary dense input proves action parity and cost, not a legal
autonomous estimator.

## 7. Single external reopen constraint

Reopen this route only with one sealed `FUSED-K5-3D-ACTION` certificate that
does all of the following without targets:

1. defines a covariance-resummed ReLU birth for both `[311]` and `[221]`, with
   a uniform error/tail law on a source-certified correlation domain;
2. contracts that birth directly with the complete retained-state future
   action, not the B1A product-of-marginal-gates mean chain;
3. proves exact dense parity at small width for source, full-state VJP, and
   endpoint action;
4. certifies effective joint source/action rank `r<=13` over all 15 layers, or
   gives a different fused circuit whose complete integrated count, including
   source evaluation and the existing host, is below `2^41`;
5. prices cache construction, symmetry, all outputs, and every collision
   multiplicity; and
6. shows a nonzero `[311]` or `[221]` witness with diagonal K5 and K22 zero.

Without that certificate, B1B remains blocked/closed-this-form.  A mean-only
rank-1024 contraction, a bivariate-only rerun, a relabeled forward Duhamel sum,
or an independent-base grade-three reset does not satisfy the constraint.

## Bound sources

- B1A `RESULT.json` —
  `785d789a9ce588989f6ab070d84c6dfdf8ec4faaacc3eabf1bbe791fb49b70e6`
- B1A terminal receipt —
  `cc5c5b094c2594297ff311aac5f366362c9d10a25833d53f54baf4f51910aa68`
- B1A independent audit receipt —
  `450378882b0fe75964cc1378fdd17b52d320ef5f80aba3ddd9bf1bb83cb761cb`
- exact ledger implementation —
  `c74c92862a3afad7e44a077c80f87fdd09c66be4f02718db24a21ea7b5f4a581`
- exact ledger result —
  `f14b4964ab6d7da4917d4c4a77c08099477a840849819cc61ee46da0d842c938`
- PC-TT5 terminal design audit —
  `ddfdfc64dec12efb5dbc7b96384e6087bc0eaad19a28a44e9d2eccd2839c16e6`
- independent IG-LC3 constraint —
  `d6275c2ace74294c4d9c6fcc6b5126dbbb9725e4dac51aeca1affba0e98c7cba`

