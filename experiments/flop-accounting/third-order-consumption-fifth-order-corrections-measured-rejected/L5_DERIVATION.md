# The correct L5-class (kappa5-free fifth-order) correction to the bivariate pair moments
## in the estimator's own Edgeworth-product-over-Mehler convention

**Question.** The estimator's pair moments use density weights `1 + L3 + L4 + L3^2/2` over Mehler
kernels. What are the exact next-order terms that use only the cumulants the host already supplies
(kappa3 diag + k21; kappa4 diag + k31 + k22) — the `L3*L4` and `L3^3` class — with coefficients in
the estimator's own normalization? And does the lambda3 shell implement them correctly?

**Answer up front.** The correct class is `Lambda3*Lambda4` (exp cross term, coefficient exactly 1,
channels (a,b) with a+b = 7) plus `Lambda3^3/6` (channels a+b = 9), with per-channel weights equal
to convolutions of the estimator's own normalized L3 block `(k03/6, k12/2, k21/2, k30/6)` and L4
block `(k04/24, k13/6, k22/4, k31/6, k40/24)`. The lambda3 shell
(`../public_lambda3_shell_probe_v1/estimator_x.py`) implements EXACTLY these terms — verified
symbolically (Section 4) and numerically to machine precision (`l5_verify.py` W-test). Its failure
on the current host is therefore NOT term error. The measured reason (Section 6): at this order
the kappa5-free terms and the kappa5 term `Lambda5` form a nearly cancelling pair on skewed
ReLU-like states, so shipping the kappa5-free half alone overshoots — it degrades exactly when the
consumed kappa3/kappa4 are accurate, which is the current (fixed) host.

---

## 1. The estimator's convention, read off the code

From `_q24_context`, `_g_group`, `_edgeworth_weights`, `_DERIVATIVE_CHANNELS`,
`_EDGEWORTH_DENOMINATORS` in `../public_q24_native_transport_energy_final_workspace_v4/estimator.py`:

```
pair moment model:   M_pq[i,j] = sum_over_channels (a,b)  W_ab[i,j] * G_ab^{pq}[i,j]
Mehler kernel:       G_ab^{pq}[i,j] = sum_{t=0}^{24} (rho_ij^t / t!) *
                                      c_i[p, a+t]/sd_i^a * c_j[q, b+t]/sd_j^b
channels:            (0,0);  L3: (0,3),(1,2),(2,1),(3,0);  L4: (0,4),(1,3),(2,2),(3,1),(4,0);
                     L3^2/2: (0,6)..(6,0)
weights:             W_00 = 1;  L3/L4 rows = kappa_ab / (6,2,2,6,24,6,4,6,24);
                     L3^2 rows = 0.5 * (self-convolution of the normalized L3 block)
```

`kappa_ab` = joint pre-activation cumulant with `a` indices on unit i and `b` on unit j (raw
units; the `1/sd^a` in the kernel makes the product the standardized cumulant — below we write
everything standardized, which is identical algebra since products of normalized blocks carry
`sd^{a1+a2} = sd^a`).

## 2. Why that convention is the operator-exponential Edgeworth — and what order 5 is

For the pre-activation pair `h = (h_i, h_j)` with Gaussian base `N(mean, Sigma)` (Sigma = the
covariance the Mehler rho comes from) and exact joint cumulants `kappa_r` (r >= 3), the formal
density is the operator exponential

```
p(h) = exp( sum_{r>=3} Lambda_r ) phi_Sigma(h),
Lambda_r = sum_{a+b=r} [ C(r,a) kappa_ab / r! ] (-d/dh_i)^a (-d/dh_j)^b .           (2.1)
```

Integrating `x_i^p x_j^q` against a `(-d)^{(a,b)} phi_Sigma` term by parts gives
`E_Sigma[ (x^p)^{(a)}(h_i) (x^q)^{(b)}(h_j) ]`, and the Gaussian-pair expectation of derivatives
expands by Mehler + Stein (`E[f^{(a)} He_t] = E[f He_{t+a}]`) into exactly the kernel `G_ab` above.
Expanding the exponential and grading by tilt order (Lambda_r has tilt order r-2):

```
order 1:  Lambda3                                    -> the L3 block      (weights kappa_ab C(3,a)/6)
order 2:  Lambda4 + Lambda3^2/2                      -> L4 + L3^2/2       (C(4,a)/24; 0.5*conv)
order 3:  Lambda5 + Lambda3*Lambda4 + Lambda3^3/6                                              (2.2)
```

Operators commute (constant coefficients), so `exp` gives `Lambda3*Lambda4` with coefficient
exactly 1 (from (Lambda3+Lambda4)^2/2! twice) and `Lambda3^3` with 1/3! = 1/6. (2.2) is complete:
there is no other kappa5-free monomial at tilt order 3. The channel weight of a product of
operators is the convolution of their normalized blocks, because derivative orders add.

## 3. RESULT: the exact L5-class weight tables (standardized cumulants)

Normalized blocks (the estimator's own): `w3 = (k03/6, k12/2, k21/2, k30/6)` indexed by a = 0..3,
`w4 = (k04/24, k13/6, k22/4, k31/6, k40/24)` indexed by a = 0..4.

### 3a. `L3*L4` — eight new channels (a,b), a+b = 7, weight = (w3 conv w4)[a]

```
(7,0): k30 k40/144
(6,1): k30 k31/36  + k21 k40/48
(5,2): k30 k22/24  + k21 k31/12 + k12 k40/48
(4,3): k30 k13/36  + k21 k22/8  + k12 k31/12 + k03 k40/144
(3,4): k03 k31/36  + k12 k22/8  + k21 k13/12 + k30 k04/144
(2,5): k03 k22/24  + k12 k13/12 + k21 k04/48
(1,6): k03 k13/36  + k12 k04/48
(0,7): k03 k04/144
```

### 3b. `L3^3/6` — ten new channels (a,b), a+b = 9, weight = (1/6)(w3 conv w3 conv w3)[a]

```
(9,0): k30^3/1296
(8,1): k21 k30^2/144
(7,2): k12 k30^2/144 + k21^2 k30/48
(6,3): k03 k30^2/432 + k12 k21 k30/24 + k21^3/48
(5,4): k03 k21 k30/72 + k12^2 k30/48 + k12 k21^2/16
(4,5): k03 k12 k30/72 + k03 k21^2/48 + k12^2 k21/16
(3,6): k03^2 k30/432 + k03 k12 k21/24 + k12^3/48
(2,7): k03^2 k21/144 + k03 k12^2/48
(1,8): k03^2 k12/144
(0,9): k03^3/1296
```

### 3c. The `Lambda5` completion (NOT kappa5-free; needs a kappa5 supply the host lacks)

Six channels (a,b), a+b = 5, weights `C(5,a) kappa_ab / 120`:

```
(5,0): k50/120   (4,1): k41/24   (3,2): k32/12   (2,3): k23/12   (1,4): k14/24   (0,5): k05/120
```

(2.2) says the FULL tilt-order-3 correction is 3a + 3b + 3c together; 3a + 3b alone is the best
kappa5-free truncation, and its residual is exactly the Lambda5 share plus tilt-order-4 terms.
All three tables apply to every moment family (M11, M21, M31, M22) with the same weights — one
density, different test functions — which is how both the base code and the shell wire it.

## 4. The lambda3 shell implements exactly these terms

Read against `estimator_x.py` (`_edgeworth_weights` extension, weights[17..34]):
- Its `L3*L4` rows are the literal convolution `w3 conv w4` with coefficient 1 — identical to 3a
  (`w22s` is a plain alias of `weights[7] = k22/4`; no symmetrization, no deviation).
- Its `L3^3` rows are `(1/3) * conv(halfL3sq block, w3)` where the half block is
  `0.5*(w3 conv w3)` = weights[10..16], hence `(1/6) (w3 conv w3 conv w3)` — identical to 3b.
- Channel bookkeeping is the canonical extension: `_DERIVATIVE_CHANNELS` += (0,7)..(7,0),
  (0,9)..(9,0); `_Q24_ORDER_GRID` rows 7..9 (consuming c[q] to 9+24 = 33 <= 38, so the frozen
  c-table already covers it); `inverse_sigma` extended to 1/sd^9 with correct powers
  (inv7 = inv6*inv1, inv8 = inv4^2, inv9 = inv6*inv3); `_Q24_TERMS` unchanged at 25; weights
  arena 17 -> 35; both consumption loops run to 35. The only other diff hunk is a
  Cholesky-first fast path in covariance repair (unrelated to term selection).

`l5_verify.py` (test W) evaluates my Section-3 tables and a literal transcription of the shell's
weight arithmetic on random asymmetric cumulant inputs: identical to < 1e-15 relative.

**Conclusion: the shell's term selection and coefficients are EXACTLY the first-principles
expansion. Its +5-11% degradation on the current host cannot be term error.**

## 5. Numerical verification of the physics (l5_verify.py)

Construct a bivariate non-Gaussian pair with exactly known cumulants and exactly computable
`E[x_i x_j]`:

```
h_u = m_u + s_u * ( g_u + eps*gamma2_u*He2(g_u) + eps^2*gamma3_u*He3(g_u) ),   u in {i,j},
(g_i, g_j) standard bivariate Gaussian, corr rho_g.
```

- Exact joint moments (p+q <= 5) by 2D tensor Gauss-Hermite (polynomial integrand -> quadrature is
  exact); exact joint cumulants via truncated 2D series log of the MGF; standardized on the pair's
  own (mean, sd, rho). Nothing is estimated.
- Exact `E[x_i x_j] = E[relu(h_i) relu(h_j)]` by root-split nested Gauss-Legendre over (g_i, g_j)
  (the same kink-split instrument class as verify.py, ~1e-12).
- The `eps^2` scaling on He3 gives clean CLT grading — kappa3 ~ eps, kappa4 ~ eps^2,
  kappa5 ~ eps^3, kappa6 ~ eps^4. (The dangerous-looking kappa6 ~ eps^2 term
  kappa(He2,He2,g,g,g,g) vanishes by leg counting: 8 legs = 4 bonds cannot connect 6 vertices —
  the same theorem that kills kappa1111 at order 2 in DERIVATION.md.) Hence, in eps:
  base-model residual ~ eps^3; + (3a+3b) removes the kappa5-free share; + 3c too -> residual ~ eps^4.
- Model evaluation replicates the code convention exactly (channels, weights, t <= 24, c[1,q] via
  the Stein closed forms; Mehler truncation checked t24 vs t30).

Report: per config and eps: truth, base residual, residual after +L3L4+L3^3/6 (deployable),
residual after also +Lambda5 (completion), the predicted-vs-measured third-order term value, and
log-log slopes. See L5_VERIFICATION.md / L5_FORMULA.json for the tables.

## 6. The measured mechanism: the kappa5-free half is one side of a cancelling pair

The verification (L5_VERIFICATION.md) shows, on generic skewed pairs, that the two halves of the
tilt-order-3 class are LARGE AND NEARLY OPPOSITE: at eps=0.32 in config C1 the L3L4+L3^3/6 shift
is -1.48e-2 while the Lambda5 shift is +1.48e-2 (net 0.2% of either); the near-cancellation holds
down the whole ladder. Consequences, all measured:
- Shipping the kappa5-free half ALONE (what the shell does) makes the pair moment residual
  2-10x WORSE than the untouched base on both generic configs (e.g. C1 eps=0.01: base residual
  +3.4e-8, +L5class-only +3.7e-7, full class +5.0e-9).
- With the pair's marginal kappa5 tuned to zero (config C3), the SAME kappa5-free terms alone are
  7-8x BETTER than base — the terms are exact; whether they help is decided entirely by the
  state's kappa5.
This is a complete mechanism for the shell's history: on the kappa4-DEFICIENT host the consumed
kappa4 entered the L3L4 weights linearly, so the one-sided shift was undersized and behaved as a
small patch co-adapted to that state's other errors; on the fixed host (exact kappa3, kappa4 at
0.82-0.94 of oracle) the one-sided shift attains full strength with no Lambda5 to cancel it —
systematic overshoot, the observed +5-11%. The near-total cancellation also means that even a
perfect kappa5 supply would buy little at this order for ReLU-like skew geometries: the right
default is to ship NO order-5 pair correction without kappa5, rather than half of one.
Secondary (untested) consistency notes: the shell also leaves `_univariate_edgeworth` (marginals,
standardization, connected resets) at tilt order 2 while pair planes go to order 3; and at large
tilts (k40 ~ 3.6 rows) even the complete exact class degrades pointwise — ordinary asymptotic-
series behavior, visible in the large-eps rows.
