# CP1792 plus nonlinear-residual RQMC: derivation and boundary

Status: **resource-feasible only at the 0.1 floor; impossible at the unchanged
CP1792 comparison bill; scientific variance gain unopened.**

This is an additive, target-free feasibility witness.  It does not read a
dataset row or target, fit a coefficient, compute a score, invoke a one-use
gate, or modify an existing CP, QMC, observer, graph, or HTML artifact.

## 1. Why `CP + QMC(network - CP)` is not a control variate

Let `f_l(X)` be the true layer-`l` activation vector and let `A_l` be the
deterministic CP1792/Hermite reported mean.  The tempting expression

```text
A_l + Q_N[f_l(X) - A_l]
```

is exactly `Q_N[f_l(X)]`; the CP value cancels.  It has the variance of the
sampling lane and cannot correct the CP bias more cheaply merely because the
subtracted constant is accurate.

More generally, an unbiased control must provide a *random* surrogate `S_l`
whose expectation is known.  If it is centered around CP,

```text
S_l(X) = A_l + q_l(X) - E[q_l(X)],
```

then

```text
A_l + Q_N[f_l - S_l]
  = E[q_l] + Q_N[f_l - q_l].
```

This cancellation theorem is the central closure result.  The CP constant can
be the surrogate's advertised mean, but it cannot itself reduce the variance
of an unbiased estimator.  A stochastic, coupled `q_l` is indispensable.

The current CP3 state contains approximate cumulants in neuron coordinates;
it is not a pathwise map from the original Gaussian input with a certified
expectation.  Treating a draw from an approximate Edgeworth law as though its
expectation were exactly `A_l` would therefore introduce an unbounded closure
bias.  This witness does not do that.

## 2. Exact-mean nonlinear surrogate used by the witness

The witness adds the smallest useful pathwise object to the unchanged
Hermite-10/CP1792 host.

Let `V` be four fixed orthonormal Walsh columns in `R^(1024 x 4)`, and define
`T = X V`.  For ideal Gaussian `X`, every row of `T` is exactly `N(0,I_4)`.
The Hermite host supplies its preactivation mean `mu_l` and Gaussian gate
`g_l = Phi(alpha_l)`.  Starting with `R_0 = V^T`, propagate

```text
B_l = R_(l-1) W_l
R_l = B_l diag(g_l).
```

The nonlinear ridge proxy is

```text
G_l(X) = mu_l + T B_l
q_l(X) = relu(G_l(X)).
```

Each coordinate of `G_l` is a univariate Gaussian with variance
`v_li = sum_k B_l[k,i]^2`, regardless of correlations between output
coordinates.  Therefore its mean is known in closed form:

```text
qbar_li = sqrt(v_li) phi(mu_li/sqrt(v_li))
          + mu_li Phi(mu_li/sqrt(v_li)),              v_li > 0
qbar_li = relu(mu_li),                                v_li = 0.
```

The implemented CP-centered surrogate and estimator are

```text
S_l = A_l + q_l - qbar_l
estimate_l = A_l + Q_N[f_l - S_l].
```

Thus `E[S_l]=A_l` and the estimator is unbiased under the ideal randomized-QMC
law.  At layer 1 the true preactivation is itself exactly Gaussian, so the
implementation reuses `q_1=f_1` and `qbar_1=A_1`; its sampled residual is
identically zero.  Layers 2--16 use the four-ridge response above.

The two lattice blocks have sizes 128 and 32 and independent continuous random
shifts.  Bit-reversed indices and odd multipliers make every coordinate of
each `2^m` block a permutation of its one-dimensional grid.  Averaging all 160
points remains unbiased because every point is marginally uniform under its
block shift.

### Numerical input-law boundary

As in the existing Phase 2 RQMC comparator, the inverse-normal argument is
clipped to `[2^-32, 1-2^-32]` to exclude infinities.  Consequently the executed
source is exactly unbiased for the clipped input law, while the ideal
unclipped statement has a controlled input perturbation.  For one standard
normal coordinate, if `b=Phi^-1(1-2^-32)`,

```text
E |Z - clip(Z,-b,b)| = 2 [phi(b) - b 2^-32]
                       = 7.131035538269621e-11.
```

ReLU is one-Lipschitz, so a ridge proxy's local mean perturbation is at most
this quantity times the response column's `L1` norm.  No deep-network output
bias bound is claimed: that would require a network-specific Lipschitz
certificate.  The receipt and verifier keep this distinction explicit.

## 3. Exact comparison-bill obstruction

The frozen current host already costs exactly
`214,122,341,378` flopscope-0.12 FLOPs.  Keeping that host unchanged leaves
zero FLOPs at the exact comparison bill.

Even one true width-1024/depth-16 sample requires at least the dense matrix
multiplications

```text
16 * 1024 * (2*1024 - 1) = 33,538,048 FLOPs.
```

So unchanged CP1792 plus one residual sample is at least
`214,155,879,426`, already `33,538,048` above the comparison bill before RNG,
ReLU, reduction, or surrogate work.  Fusing calls cannot change flopscope's
additive operation count.  Reducing the cap or deleting CP work could create
room, but that would no longer be the stipulated current CP1792 host.

Therefore an unbiased nonconstant residual hybrid is **closed at the exact
214.122B comparison bill**.

## 4. Official 0.1-floor resource result

The inclusive 0.1-floor ceiling is `floor(2^41/10) = 219,902,325,555` FLOPs.
The official dataset-free subprocess meter measured:

| Quantity | Measured | Gate / comparison |
|---|---:|---:|
| flopscope 0.12 FLOPs | 219,870,793,378 | <= 219,902,325,555 |
| margin to floor | 31,532,177 | positive |
| increment over CP1792 | 5,748,452,000 | exact-comparison FAIL |
| residual wall time | 0.1050503752 s | < 0.4 s |
| sampled descendant peak RSS | 2,256,158,720 B | < 8 GiB |
| output | finite float32 `(16,1024)` | PASS |

The measured increment closes operation-by-operation.  Its dominant entries
are `5,671,418,624` matrix-multiplication FLOPs and `27,197,440` inverse-normal
FLOPs; all remaining operations total `49,835,936` FLOPs.

Authoritative repeatable synthetic run:

`../meter_protocol/runs/cp1792_rqmc_nonlinear_residual_n160_r4_n1_s1000_seed20260823/receipt.json`

The common retained-receipt verifier returns
`PASS_METER_RECEIPT_VERIFICATION`.  No target-bearing execution was performed.

## 5. Scientific boundary

This is a mathematical and resource witness, not a promoted estimator.

- The expensive CP3 correction cancels algebraically from the unbiased
  estimator.  Only the host-derived stochastic response can reduce variance.
- Four Walsh ridges span `4/1024 = 0.390625%` of the input space.  Their
  correlation with layers 2--16 is not certified by the analytic state.
- Only 160 fine samples fit beside CP1792.  No target-free theorem says their
  residual variance beats CP1792's closure error or an equal-bill sampling
  lane.
- No variance result, target metric, score, or selection was run here.

Accordingly the honest conclusion is:

1. **Exact CP1792 bill:** infeasible by additive cost closure.
2. **0.1 floor:** a mathematically valid nonlinear-residual RQMC source fits
   all measured resource gates.
3. **High-leverage claim:** HOLD.  The CP constant cancellation means a serious
   successor should either derive a certified pathwise chaos surrogate from
   the CP state or omit the CP3 transport and spend those FLOPs on the
   stochastic residual.  Neither change is silently made in this artifact.
