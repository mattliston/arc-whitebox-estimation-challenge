# FB-EC4-32 accuracy-frontier preregistration

Status: **target-free design only; no estimator, execution, target open, score,
fit, sampling, submission, or graph mutation is authorized here**.

This package specifies the smallest coherent successor ladder to run only if
the final byte-frozen V3 parent passes its Phase-2 source, parity, literal
meter, finiteness, memory, wall-time, residual-time, and actual-moment gates,
but its preregistered raw MSE is not below `1e-8`.

## 1. Parent and non-negotiable invariants

The scientific parent is FB-EC4-32 + CP4R2048:

- cap-16384 full-birth symmetric CP3;
- dense cubic coordinate-star residual H3;
- coherent `Star4(H4) + alpha P_C` quartic residual;
- cap-2048 signed symmetric CP4 history;
- diagonal-preserving correlation-spectrum PSD repair;
- pair Edgeworth `1 + L3 + L4 + 1/2 L3^2` at Q32; and
- marginal-only W16 specialization.

Every successor must preserve the same input law, exact h1 construction,
float32 carrier policy, float64 mean/covariance/consumer policy, layer order,
connected-moment conventions, support guards, final specialization, and
Phase-2 execution contract.  A component may be removed only as a whole
tensor or whole asymptotic term.  In particular, no arm may zero K31 while
retaining an independently inconsistent K4/K22 plane, clip raw3/raw4, shrink
correlations globally, tune a depth coefficient, use targets during
construction, or introduce sampling or fitting.

The V3 source observed while this design was written is explicitly mutable
and is not a parent freeze.  No successor source may be derived until a later
V3 manifest binds the exact estimator bytes that passed all prerequisite
gates.

## 2. Primary nested ladder

`A0_FULL` is the eventual frozen V3 reference.  The three first-priority
successors are nested, so every adjacent comparison removes exactly one
scientific state family.

### A1: CP4 history off

Keep CP3, H3, coherent H4+alpha, Q32, the full Edgeworth consumer, and the PSD
repair.  Remove the CP4 factors, push, preconsumer provider add, nonlinear
gate/birth/truncation, retained-provider subtraction, and W16 CP4 read.  At
every reset H4+alpha is fit directly to the complete corrected `(K4,K31,K22)`
planes.

`A0_FULL - A1_NO_CP4` isolates the complete CP4 carrier and tensor-completion
history mechanism, including dense-ray retention, birth, top-energy
truncation, and stored-float32 rounding.  The visible reset `K4/K31` targets
remain represented in both arms; the changed object is whether prior dense
quartic rays survive beyond the coordinate-star residual under that complete
history policy.  This is the first arm because historical coherent CP4
evidence contains a late sign reversal.

### A2: all carried fourth order off

Starting from A1, remove H4 and alpha as one coherent tensor component and set
the affine `(K4,K31,K22)` input to zero.  Remove `L4` from both univariate and
pair consumers, but retain `1/2 L3^2`: the latter is a fourth asymptotic-order
term generated entirely by the retained third cumulant.  Compute only the two
pair moments needed by the remaining state, `(M11,M21)`; omitting unused
`(M31,M22)` is an exact execution specialization, not a scientific change.

`A1_NO_CP4 - A2_THIRD_ONLY` isolates the coherent H4+alpha carrier together
with the matching L4 consumer contribution.  It never tests the previously
falsified independently scheduled K31 plane.

### A3: H3 residual off / CP3-only third state

Starting from A2, remove H3 reset, affine push, and final diagonal read.  CP3
is the sole third-cumulant tensor and still supplies both K21 orientations and
the final K3.  Because no post-ReLU K21 residual is retained, compute only
`M11`; omitting unused `M21` is exact specialization.

`A2_THIRD_ONLY - A3_CP3_ONLY` isolates the dense coordinate-star H3 residual.
The cap-16384 CP3 state is not truncated at this depth, so a reversal points
to bias from H3's source-only all-distinct completion rather than lost CP3
rank.

## 3. Conditional consumer arm

`A4_NO_L3SQ` is built only after A0--A3 have one fixed local score, and is
applied to the unique best carrier host selected by the frozen rule below.
Remove all seven total-derivative-order-six pair channels and every
univariate `K3^2` term.  Keep Gaussian, L3, and L4 terms unchanged.  This is
the coherent first-order-in-cumulants Edgeworth truncation
`1 + L3 + L4`, not a coefficient fit.

`winner - A4_NO_L3SQ` isolates the formal `1/2 L3^2` contribution under the
best already-selected state.  A win for A4 means the squared-skew term is
amplifying error in the approximate recursive K3/K21 state; it does not
license tuning its coefficient.

No other accuracy variant is prioritized before these four.  In particular,
no CP4 cap sweep, Q sweep, PSD-floor sweep, alpha schedule, layer band, or
coefficient sweep is allowed.

## 4. CP3, Q32, and PSD separation without incoherent arms

### CP3 anchor

CP3 already has a sealed Phase-2 anchor: the Gaussian/Hermite-order-4 host has
final raw MSE `3.849300358689406e-6`, while the cap-16384 marginal CP3 arm has
`3.5989996088739297e-6`; recursive reported-mean carry reaches
`9.340196551564291e-7`.  These do not replace the A2/A3 same-host H3 contrast,
but they establish that CP3 itself is not an untested ingredient.  A same-host
Gaussian arm is opened only if A3 is worse than the frozen Gaussian anchor or
the A2/A3 contrast is directionally inconsistent across layers.  It must
remove CP3 as a whole; it may not retain a K3 diagonal side channel.

### Q32 numerical-order diagnostic

Before scoring a Q variant, replay the selected carrier host target-free at
Q24 and Q32 with the identical Q38 coefficient generator and all 17 derivative
channels.  Only the rho-power/inverse-factorial truncation changes.  Record
prediction disagreement, covariance/K21/quartic state disagreement, and the
first layer at which it exceeds floating-point parity.  Score Q24 only if its
final prediction-disagreement MSE exceeds
`0.01 * min(A_selected_final_raw_mse, 1e-8)` or it changes a moment-feasibility
verdict.  Q above 32 remains closed: the frozen older-host R40/R80 diagnostic
put series-order change at at most `8.55e-9` relative against a `3.09e-2`
one-step error, while the present full-budget source has only about 19.3B
measured headroom before the V3 rewrite.

### PSD intervention diagnostic

There is no PSD-off accuracy arm.  An indefinite covariance is not a coherent
law and cannot diagnose the repair.  At every reset record the raw correlation
minimum eigenvalue, number and sum of eigenvalues changed by the floor,
relative Frobenius repair mass, maximum off-diagonal change, and downstream
state delta.

Only if the spectral repair is active at a material layer may one lawful
comparator be built.  The comparator is frozen correlation-ridge repair

```text
epsilon = 2^-40
tau = max((epsilon - lambda_min(R)) / (1 - epsilon), 0)
R_ridge = (R + tau I) / (1 + tau).
```

Because raw `diag(R)=1`, this has exact unit diagonal and eigenvalues
`(lambda_i+tau)/(1+tau) >= 2^-40`.  It is a different coherent PSD
policy, not a no-repair or global ad-hoc shrink arm.  No floor or ridge sweep
is allowed.

## 5. Exact evaluation order and selection

1. Freeze and independently audit V3.
2. Require one common-driver target-free meter with FLOPs `<=2^41`, residual
   `<=0.4 s`, wall `<120 s`, descendant RSS `<8 GiB`, and finite float32
   `[16,1024]` output.
3. Require the actual post-Edgeworth univariate Stieltjes/Hankel diagnostic to
   pass without scalar clipping.
4. If A0 is below `1e-8`, stop the ablation ladder and promote A0.
5. Otherwise build A1--A3 additively from the exact frozen A0 bytes, audit
   one-factor source diffs and tiny algebra, then meter each with the identical
   common driver.  A projected cost is never a meter PASS.
6. Freeze A0--A3 as one roster before one authorized local target open.  Report
   final/all-layer raw MSE, all 16 layer MSEs, per-network paired final deltas,
   and wins.  No result-dependent source change occurs during this open.
7. Select the lowest final raw MSE.  If two arms differ by less than
   `1e-10` final raw MSE or neither wins at least 55 of 100 paired networks,
   retain the simpler/cheaper arm.  Layer 1 may not regress beyond the frozen
   reference-noise receipt; layer 2 may not regress by more than `1e-8`
   absolute raw MSE.
8. If the selected arm remains above `1e-8`, build and independently audit the
   single A4 consumer arm.  The Q24 and PSD comparators remain conditional on
   their target-free gates above.
9. Only a locally promoted, fully metered, independently audited source is a
   leaderboard candidate.  Known-failed, duplicate, nonfinite, over-budget,
   or target-adapted arms never consume a board attempt.

The primary causal reading is adjacent, not all-pairs: A0/A1 is CP4 history,
A1/A2 is coherent H4+alpha/L4, and A2/A3 is H3.  CP3 is read against its frozen
Gaussian anchors unless the conditional same-host Gaussian arm is triggered.

## 6. Resource statements

The authoritative V2 full-source meter is `2,179,715,874,872` FLOPs.  V3 has
no authoritative bill at the time of this design.  Component arithmetic is
partitioned as follows:

| contrast | exact dominant arithmetic removed | status of full delta |
|---|---:|---|
| A0 -> A1 CP4 off | `233,997,074,432` | CP4 audit projection including enumerated lower work is `234,507,794,432`; literal meter required |
| A1 -> A2 H4/L4 off | `182,446,981,120` H4 GEMMs + `97,860,116,480` recurrent Q32 pair einsums + `411,176,960` h1 Gaussian pair einsums = `280,718,274,560` | lower-order and packing delta requires literal meter |
| A2 -> A3 H3 off | `122,346,799,104` H3 GEMMs + `48,930,058,240` recurrent Q32 pair einsums + `205,588,480` h1 Gaussian pair einsum = `171,482,445,824` | lower-order and packing delta requires literal meter |
| winner -> A4 no L3-squared | `80,590,684,160` Q32 pair einsums | univariate/weight/packing delta requires literal meter |
| Q32 -> Q24 diagnostic | `48,082,370,560` pair-einsum arithmetic | rho-feature/stack delta requires literal meter |
| spectral PSD block | `330,683,121,664` for 14 fp64 eigensolves plus reconstructions | ridge-comparator delta requires its own meter |

“Exact dominant” means the integer FlopScope 0.12 arithmetic for the named
fixed-shape kernels, checked against V2's operation receipt.  It is not called
an exact whole-estimator meter delta.  The whole-estimator delta is exact only
after both immutable parent and child receipts exist and their integer bills
are subtracted.

## 7. Expected causal interpretations

- A0 wins early and late over A1: the complete CP4 carrier/history mechanism,
  including birth, truncation, and rounding, is useful.
- A1 wins late after A0 wins early: CP4 cap/energy history becomes stale or
  cancellation-dominated; keep coherent H4+alpha and drop CP4.
- A1 wins over A2: the coherent H4+alpha/L4 family carries useful fourth-order
  action.  A2 winning closes that complete fourth-order host, not K31 alone.
- A2 wins over A3: H3's reset-complete K21 plane is useful.  A3 winning means
  H3's coordinate-star all-distinct continuation is biased.
- A4 wins: the formal `L3^2/2` term is harmful under approximate transported
  third cumulants; do not tune its coefficient.
- Q24/Q32 target-free parity with large raw error: numerical series order is
  exonerated and the error is structural.
- negligible PSD intervention with large raw error: PSD repair is exonerated.
  Material intervention plus a ridge win localizes covariance-repair policy.

These readings are diagnoses of this exact host.  They do not authorize a
sampler, learned residual, coefficient fit, or universal tensor-rank claim.
