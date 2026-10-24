# Coupled stochastic residual: source-only scope and feasibility audit

Recorded 2026-09-06 PDT (clock: 2026-09-07 00:22:30 UTC). Phase2 only.
No estimator execution, target-array decoding, fit, new sampling, metering,
submission or forum action. Existing sources and receipts are unchanged.

## Decision

**No new executable mechanism is substantiated by this audit.** Exact-input
Stein controls and nonlinear coupled surrogates are not universally closed.
But neither an actual cheap high-capture surrogate nor the required variance
certificate has been supplied. Calling the old four-Walsh witness
"unmeasured" is correct; treating that fact as a positive reason to run it is
not. Its frozen independent audit also explicitly withholds qualification.

The nearest mathematically viable branch is the already-audited final
preactivation Hermite control with independently certified true-law moments.
The missing moment provider, not its cheap polynomial arithmetic, remains
the unresolved state-acquisition problem. No new provider is identified here.

## 1. The exact budget is a residual-variance budget

For a deterministic, samplewise surrogate `s(X)` with a genuinely known mean,

```
mu_hat = E[s] + (1/N) sum_a (f(X_a) - s(X_a)).
```

For independent Gaussian samples, let `v_R` denote mean per-output residual
variance, `c` the complete cost per coupled sample, and `C0` setup cost.
Then expected final MSE is `v_R/N` and adjusted MSE is

```
(v_R/N) max(0.1, (C0 + N*c)/2^41).
```

Thus, above the floor, the sample term is
`v_R * (c + C0/N) / 2^41`. Increasing N alone cannot overcome an excessive
variance-times-path-cost product. Approximate supplied control means add the
uncancelled squared bias of their combined error.

Under the DECLARED classical dense convention `c=2^25=33,554,432` for the
16 matrix-vector layers, even ignoring setup, RNG, activation and control
work, adjusted `<1e-9` needs `v_R < 6.5536e-5`. Relative to the archived
rank35 final activation variance `.06679448554`, that is capture `>99.9019%`.
One extra dense 1024-by-1024 control application tightens it to `6.16809e-5`.
These are arithmetic yardsticks, NOT lower bounds on every possible fast
matrix-multiplication sampler. A proposed accelerated implementation must
supply its own complete certified cost.

For randomized QMC, substitute the actual randomized-rule variance for
`v_R/N`. IID formulas are not rigorous lower bounds on RQMC. None of the
retained resource-only receipts measures that rule variance.

The accurate bank constant provides no free stochastic accuracy:

```
A + Q[f - (A + q - qbar)] = qbar + Q[f - q].
```

If `qbar` is only an approximate mean, its error survives. If `qbar` is
estimated by the same samples of q, the control cancels. Independent,
full-cost centering does not supply a free oracle; in the orthogonal
control/residual case, optimal splitting has variance proportional to
`(sqrt(v_R)+sqrt(v_s))^2`, not `v_R` alone. Offline/per-network fitting and
independent mean acquisition must also be priced.

## 2. What Phase2 actually executed or left unqualified

Paths below are relative to the Phase2 experiment directory.

| Source/receipt | Actual scope and result | What it does not close |
|---|---|---|
| `public_quadrature_derivation_v1/{realnet_rules.py,realnet_rules.json,realnet_measurements.json,nnls_v3.json}` | Rank35 MC, paired affine CV, antithetics, two-direction GH/MC and shifted lattice; cross-net variance/affine checks on36/120/234. At4096 paths, MC MSE1.8942e-5, affine1.4422e-5, antithetic1.5834e-5, GH2 .00101067. Paired affine capture .20505; coefficient optimization .20632. | Arbitrary nonlinear exact-mean controls or every possible conditional integration rule. The measured gradient-subspace captures are specific basis diagnostics, not a supremum over all subspaces. |
| `cp1792_rqmc_nonlinear_residual/{DERIVATION_AND_BOUNDARY.md,INDEPENDENT_TARGET_FREE_AUDIT.json,estimator.py}` | Four fixed Walsh latent directions; shifted ReLU proxy at every layer, exact ideal-Gaussian proxy means; independent128+32 shifted-lattice blocks. Retained flopscope0.12 synthetic resource bill219,870,793,378. No scientific variance or target score. | Its exact frozen bytes are on HOLD, not a qualified untested candidate. See section3. |
| `rqmc_fullbudget_rank1024_cv/estimator.py`; `meter_protocol/runs/rqmc_fullbudget_r1024_crossfit_n57344_draft1/raw_whest_stdout.json` | Full1024-dimensional Gaussian-response shifted-ReLU proxy and leave-one-shift-out coefficient source already exist. The attempted57344-path meter failed immutable-array item assignment, with no estimator outputs or scientific result. | This is not an accuracy failure and not a novel proposed architecture. No complete passing meter or high-capture receipt was found. |
| `p2r0f3c_late_layer_hermite_cv_audit_v1/{AUDIT.md,AUDIT_RECEIPT.json}` | Target-free Gaussian residual calculation using the200-row historical Q24 alpha/sigma tape; source/bias/cost audit. Degree8 Gaussian residual variance8.24288e-5; degree12 4.41320e-5. | Actual network residual variance and true Hermite-control means are absent. This is a moment-provider constraint, not a runnable scored arm. |
| `equal_budget_sampling_comparator/PREREGISTRATION.md`; `fullbudget_sampling_primary_v1/PREREGISTRATION.md` | Exact/near-budget IID, affine-CV, Sobol and antithetic/whitened sampling resource preparations, including a64000-path source. | Synthetic plumbing metrics are not Phase2 raw-accuracy measurements. |
| `astra_phase2_reexamination_v1/SPECTRAL_SAMPLE_CORRECTION_BOUND_V2_CLOSED.md` | Current-bank byte parity, coherent covariance and existing truth means on35/100. Even optimistic per-mode oracle sample gains at8192 gave raw ratios .983697/.986034 but adjusted ratios1.291249/1.295004. | Only diagonal gains in that fixed covariance eigenbasis, not a new low-variance coupled residual covariance. |
| `public_g19_math_acceleration_v1/LANE_g3_tracker.md`, T-78 | Sampled K22/K31/variance acquisition: even65536 paths leaves K31 relative error roughly .9--1.2 and variance noise around .6%; K22 is easier. | A different explicit action-specific control with a new variance certificate. |
| `p2r0g4_coupled_stack_terminal_v1/TERMINAL_RECEIPT.json`; `p2r0g1b2_kerdock_complete_action_terminal_v1/TERMINAL_RECEIPT.json` | Existing directional/mixed action capacity and complete-cost closeouts; old scalar Markov-suffix perfect-mean-reset results. Affine controls cancel under their exact symmetric rule. | Arbitrary richer same-row nonlinear controls or a new full-budget architecture. |

The old `QUADRATURE_DERIVATION.md` universal "no tractable surrogate exists"
language is not a theorem. Its annealed kernel repeatedly centered at each
layer, unlike the actual network. The correction
`astra_phase2_reexamination_v1/NNGP_CENTERING_AUDIT_V1_CORRECTION.md`
replaces that theory while retaining the empirical failures. Its corrected
ideal degree<=64 fraction is .969303, not a complete finite-network bound.

`public_stein_psd_anchor_reaudit_v1/verify_static.py` audits a formal
moment/PSD-anchor construction, not a neural input-space Stein CV experiment.
The similarly named estimator "conditional integration contract" is an
approval gate, not Rao--Blackwell integration. Many `*_cv*` directories mean
cross-validation. No Phase1 result is needed for this conclusion.

## 3. The four-Walsh witness is not a compelling scientific reopening

The independent receipt says `HOLD_CURRENT_EXACT_BYTES` and explicitly says
it is not execution/scoring authority. It permits citing its ideal control
identity and retained resource measurement, but not executed clipped-law
exact unbiasedness or the frozen verifier's pass.

Concrete defects: the sampler clips its inverse-normal arguments while the
proxy mean is for an ideal Gaussian; these are not exactly the same law. The
first scalar ReLU counterexample differs by3.56552e-11. There is no deep
true-path clamp-bias certificate. Its frozen numerical verifier also calls
removed `np.trapz`; a compatibility shim was diagnostic, not qualification.
These are not evidence of a large raw error, but cannot be silently ignored.

The expensive CP mean cancels from the actual CV output. The surviving
nonlinear control is measurable with respect to only four fixed input
directions. Its best possible L2 residual is at least
`E Var[f(X) | V^T X]`, but that quantity was not measured for these Walsh
directions. The archive's gradient-basis conditional captures (6.5% at8,
27.7% at128,63.1% at512) are adverse plausibility evidence, not a certified
bound for this exact V.

With160 samples, the IID yardstick for raw`<1e-8` at the.1 floor is
`v_R<1.6e-6`, or approximately99.9976% capture relative to the archived
rank35 variance. There is no analytic or measured RQMC variance certificate
bridging that gap. Omitting CP to buy more paths or replacing the four
directions would change the frozen method and cannot be called replaying
this witness. I do not recommend a one-shot scientific screen of it.

## 4. New source-only obstruction: an arbitrary-width zero-bias shallow ridge CV

For any output coordinate of a shallow zero-bias ReLU surrogate,

```
s(x) = c + a^T x + sum_r b_r ReLU(v_r^T x),
s(x) - s(-x) = 2 a^T x + sum_r b_r v_r^T x.
```

Its odd part is exactly linear, regardless of ridge count. Consequently its
Gaussian Hermite projection on every odd degree>=3 is zero. An exact
first-layer ReLU vector followed by an arbitrary linear output map has this
same obstruction. The L2 residual must retain the target's complete odd
degree>=3 energy.

The archived NNLS spectrum has fitted odd mass .566128 and fitted linear
mass .224295, a difference .341833. That is a fitted empirical diagnostic,
NOT a rigorously certified lower bound: total fitted mass is1.03008 and the
fit has uncertainty. It nevertheless gives no support for the sub.001
residual fraction required by the dense-path budget. No width sweep is
justified by relabeling all first-layer features as a richer exact-mean CV.

This argument does NOT apply to shifted ridges, products of ridges, or
general Stein fields. Shifted full-rank response controls already have the
unqualified source noted above. A generic quadratic map of1024 first-layer
features for1024 outputs needs a cubic-sized coefficient object and generic
cubic work per path; a low-rank restriction would need its own high-capture
information witness, not a free exact-integrability claim.

## 5. Exact Gaussian-input Stein/martingale centering: logically open, no cheap instance

For a suitably regular Gaussian vector field `g`,
`Tg = div(g) - x^T g` has zero mean. The old
`LANE_two_pi_counterterm.md`, sections17.3--17.4, already derives this and
the formal OU inverse. Its finite-control rejection is explicitly scoped to
the available D11/AQ sources and that action budget, not all future fields.
`LANE_h4_reverse_adjoint.md`, section26, gives the same distinction.

For a smooth near-network potential `u`, taking `g=grad(u)` gives
`Tg=Delta(u)-x^T grad(u)`. Generic exact propagation of its Laplacian for all
1024 outputs requires the input-Jacobian metric, not merely the forward
activations. A direct dense implementation carries a1024-by-1024 Jacobian
and multiplies it by each W: order16*n^3 per sample rather than16*n^2,
before nonlinear Hessian terms. Hutchinson directions can reduce that bill
but introduce an additional stochastic trace residual whose variance must
be included. This is not a lower bound on every possible special field.

Using ordinary ReLU derivatives and setting its second derivatives to zero
almost everywhere is NOT a centered shortcut: the missing kink-boundary
distribution carries the Laplacian contribution. Smoothing removes that
formal defect but does not certify residual variance. Special first-order
fields may be cheap, but no specified field with adequate capture is
provided by the current bank.

Similarly, a true Doob/martingale control needs actual conditional
centering and a future map measurable before the fresh noise. See
`LANE_nonlocal_depth_kernel.md`, equationN.224 and surrounding discussion.
A deterministic Gaussian moment closure is not a samplewise fresh-noise
representation; declaring its error an innovation does not make it one.

The high-degree final-preactivation Hermite branch is closest in variance:
its degree12 *Gaussian* residual would give a dense-path-only adjusted term
about6.734e-10. But its true-law Hermite means through12, combined bias and
actual residual variance remain unknown. Replacing those with the card's
approximate moments precisely reinstates the closure error we want to
remove. This is not an unexecuted complete candidate.

## Provenance and endpoint

Filesystem mtimes, not handoff prose, place the four-Walsh independent audit
at2026-08-23 14:52:05 PDT, the full-rank CV source at2026-08-23 17:58:39 PDT,
the late-Hermite audit receipt at2026-08-26 12:06:13 PDT, and the NNGP
correction at2026-09-06 02:39:30 PDT.

This note authorizes no numerical work and creates no new arm. The narrow
open requirement is a concrete, target-law-coupled, cheaply centered random
surrogate (or explicit Stein field) with its entire acquisition/control bill
and enough residual-variance evidence. The archive neither supplies that
object nor proves that no future object can exist.
