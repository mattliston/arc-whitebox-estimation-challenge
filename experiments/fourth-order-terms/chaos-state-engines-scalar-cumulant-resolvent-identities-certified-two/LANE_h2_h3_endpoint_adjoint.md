# H2/H3 endpoint adjoint — the surviving exact route

Durable pure-mathematics lane record (fable, opened 2026-08-31 after the
V34 sentinel falsification). Derivation-first; numerics only as
preregistered checkpoints on banked panel data.

## 0. Why this lane exists (measured stakes)

- Oracle ladder (10 nets): deployed 3.197e-8 → exact v 1.466e-8 →
  +k3 1.029e-8 → +k4 9.25e-9. The architecture suffices; ingredients
  are the whole gap.
- Card ingredient accuracy (r35/r100, banked): per-unit v is already
  0.05–0.26% relative per layer (the win lives inside the last ~0.25%
  of v); k3 is 6.0–10.5%.
- Tensor-borne share: the full cum3 engine reaches k3u rel 0.036–0.068
  vs the card's 0.060–0.105 — a ~40% k3-error reduction is carried by
  content only the 1024³ tensor knows (receipts:
  ../public_g19_attribution_v1/MAP_GAP_HOST_rank0035.json + this
  session's k3 comparison on tensor_planes k3u).
- Everything LEARNED or COMPRESSED is falsified (kt −80% frozen
  transfer; aux-planes; block-means; 899-param V34 sentinel wrong-sign
  on rank35). Only EXACT computation survives.

## 1. The program

SOL's H4 reverse-adjoint result (LANE_h4_reverse_adjoint.md): the
endpoint effect of every bare H4 square birth accumulates exactly into
one r⁴ = O(n²) object with O(Ln³) work, because the even-H4→even-H4
ReLU derivative is a common scalar in the frozen second-order tangent.

Conjecture to develop (the H2/H3 extension): the endpoint-mean
functional of the FULL cum3 tensor is computable without carrying the
tensor, by the same reverse-mode structure that lets backprop avoid
materializing Jacobians:

  S[unit a] = Σ_l ⟨ T_l , A_l^{(a)} ⟩,   A_l^{(a)} ∈ R^{n×n×n}

where A_l = the adjoint (gradient) of the final mean w.r.t. the cum3
entries at layer l. The adjoint is never materialized; it satisfies a
BACKWARD recursion through the transposed Mehler/weight maps. Two
structural facts make this potentially cheap:

(a) t=0 dominance (measured 2026-08-30: the rank-1 Mehler truncation of
    the covariance-error chain keeps dv-corr 0.785 of 0.813): the
    leading adjoint factorizes through conjugation by
    A_l = diag(e1) W_{l+1}, giving A_l^{(a)} ≈ low-rank sums of
    (u⊗u⊗q)-type dyad³ terms.

(b) If rank(A_l) ≤ r (as a sum of r pure tensors with deploy-known
    factor vectors), then S = Σ_r cum3(x_rᵀz, y_rᵀz, q_rᵀz): scalar
    joint cumulants of PROJECTIONS — each exactly trackable by a
    forward low-dimensional joint-moment recursion (the K3S tracker,
    validated free-running ±1.3–2%, is the r=1, x=y=q=ŵ case).
    Cost: r × O(Ln²) ≈ free at r ≤ 100.

NOTE the difference from the failed compression lanes: nothing is
compressed or fitted. The adjoint DEFINES which exact scalars are
needed; the forward scalar recursions COMPUTE them exactly (up to
their own validated closure error). Wrong-basis and transfer diseases
do not apply — there are no learned parameters.

## 2. Known obstructions to resolve

1. Parity/order mixing: unlike H4→H4, the H2 and H3 chains leak into
   the mean channel at every layer (dE[φ]/dv ≠ 0, dE[φ]/dk3 ≠ 0). The
   adjoint must therefore carry the mixed (H0…H4) block-recursion; the
   Jordan-quotient machinery in SOL's lane is built for exactly this
   intertwining — needs the H2/H3 case worked.
2. Adjoint rank growth: each backward step through D∘ (Mehler t≥1
   terms) multiplies rank by the kernel's term count. Yesterday's
   forward analog stayed value-faithful at t=0; the backward
   truncation error must be BOUNDED, not assumed (checkpoint C2).
3. The scalar-recursion supply: cum3(xᵀz, yᵀz, qᵀz) for arbitrary
   fixed directions needs the trivariate ridge closure — the K3S
   tracker's derivation generalized from (ŵ,ŵ,ŵ). Its free-run
   accuracy for generic directions is unmeasured (checkpoint C3).

## 3. Preregistered numeric checkpoints (banked data only)

- C1 (adjoint rank): on r35, compute the exact endpoint-mean adjoint
  restricted to the t=0 chain at layers 5/9/13; measure the numerical
  rank of A_l needed to keep 90/95/99% of ⟨T_l, A_l⟩ against the
  banked engine tensor planes' known contractions. GATE: r(95%) ≤ 128.
- C2 (t-truncation): repeat C1 with t≤1 and t≤2 backward kernels;
  GATE: t≤1 keeps ≥ 90% of the t-full value (mirrors the forward 0.785
  measurement).
- C3 (tracker generality): for 16 random (x,y,q) triples from the C1
  adjoint factors, run the trivariate scalar recursion forward on r35
  and compare against the engine tensor's exact contractions.
  GATE: median rel error ≤ 10% (the K3S precedent: ±2%).
- C4 (end-to-end): assemble S for the v-channel on r35+r100 with
  frozen everything; grade through the certified consumption. GATE:
  ≥ 8% capture on BOTH nets with the SAME (unfitted) construction.

Fail any gate → record and close honestly; the remaining route is then
large-truth-set exact numerics only.

## 4. Checkpoint derivations

(appended below by the working agent)

## 2026-08-31 — C1 executed (fable): borderline PASS + the C3 sharpening

**C1 result (r35, banked data, receipts
`../public_g19_attribution_v1/C1_ADJOINT_RANK_rank0035.json`):**
the t0 endpoint adjoint per source layer l is exactly
`M_l = diag(a2_l) [ Σ_{L>l} B_{l→L} diag(S_v·g_L) B_{l→L}ᵀ ] diag(e1_l)`
with g the resid back-transport through the φ/W consumption chain.
Sanity: Σ_l ⟨dK21_l, M_l⟩ implies 15.7% linear capture — matching the
certified v-channel value through a fully independent construction;
the algebra is confirmed. Value-rank r95 per layer: 6–135 (median
~45); L9 exceptional (>200 at 90%). Signs mix across layers
(Σ|v|/|Σv| ≈ 1.2 — mild cancellation). GATE READ: borderline pass at
r ≤ 128 for all layers except L9; ~50–150 tracked reads/layer suffice.

**C3 sharpened — the real obstruction:** each rank-1 read
⟨dK21_l, x⊗y⟩ is a PARTIAL-DIAGONAL tensor contraction
Σ_i x_i cum(z_i, z_i, yᵀz), which does NOT factor through
low-dimensional projections (the 2026-08-30 obstruction). The
randomized identity rescues trackability:
  E_s[ cum(p_sᵀz, p_sᵀz, yᵀz) ] = Σ_i x_i cum(z_i,z_i,yᵀz),
  p_s = s ∘ √x (iid signs s), for x ≥ 0 (split x = x⁺ − x⁻).
Each draw IS a true projection-cumulant → trackable by the K3S-class
bivariate scalar recursion. OPEN QUESTION (decides C3): the
off-diagonal cross-term variance per draw — the number of sign draws
S needed for the read precision that survives the layer-sign
cancellations. If S ≲ 8 per read, total trackers ≈ r·S·L ≈ 10⁴ at
~1e6 flops each ≈ 0.5% of budget: billable. If S ≳ 100, the family
dies on cost like everything else.
NEXT: measure the per-draw variance directly on banked planes
(⟨K21, p_sp_sᵀ...⟩ evaluable offline for sampled draws) BEFORE
building any tracker.

## 2026-08-31 — C3 pre-gate: FAIL (randomized-read branch closed)

Measured (d2 engine, MODE3_PROBES=8, layers 6/10/14, receipts
`../public_g19_attribution_v1/TENSOR_RECURSION_rank0035_v2b.json`):
the mode-3 slices X = T·y carry off-diagonal/diagonal sum-square
ratios 11.5–117.3 (median ≈ 45) for deploy-family probe directions.
With the C1 adjoint factors' mixed signs, the diagonal read sums
INCOHERENTLY, so a single sign-draw's relative sd is ~√(2·ratio) ≈
9–10× the signal: S ≈ 10³–10⁴ draws per read, far beyond the S ≲ 8
viability bound. Aggregate-level averaging across ~10³ reads could in
principle reduce total cost to 1–5% of budget, but only if the
(unbuilt, unvalidated) trivariate trackers' closure errors
decorrelate across draws — a fragile stacked bet of exactly the class
this month falsified repeatedly. VERDICT: the randomized partial-
diagonal read branch closes. What remains of this lane: the exact
common-scalar sub-chains (SOL's H4 result and any H2/H3 analogues
under the Jordan-quotient intertwining) — pure derivation, no
randomization. The partial-diagonal read obstruction is now measured
at every level it can appear: plane compression (08-30), unit-state
learning (V34), and randomized projection reads (this entry).

## 2026-08-31 — C4v1 closed form: NULL, with the sharpest diagnosis yet

Implemented the birth-accumulation closed form (completion births,
DTOT=2 chain motifs, diagonal cdf dressing, matmul-reduced;
receipts `../public_g19_attribution_v1/C4V1_CLOSED_FORM_rank0035.json`
+ run_c4v1_closed_form.py). Result: corr(Slice_l, dK21_l) ≈ 0.00 at
every layer with the scale right (snorm ≈ 2× dknorm). Reading: the
raw accumulated birth content is common to BOTH the engine and the
host's plane recursion and cancels in dK21 = engine − host. The
tensor-borne value is the image of the births under the TRANSPORT
DIFFERENCE operator Δ = (true distinct transport) − (host's
plane-projected transport). Closed-forming ⟨A, Δ[births]⟩ requires
both transports to matching precision — the difference is small
relative to either term (hence corr 0 for the undifferenced form).
This is the exact object SOL's Jordan-quotient intertwining is built
to organize: Δ acts on the quotient of the full chain by the
plane-representable subspace. NEXT (multi-day derivation, no
shortcut): express Δ per layer in the quotient basis; determine
whether its chain-motif matrix elements admit the same matmul
reduction. The C1 adjoint machinery grades any candidate instantly.

## 2026-08-31 — Δ is a SCREENED transport (measured), and the target refines

Pre-check on banked planes (r35, L6/9/12): |dK21_kt−host| is 4–5×
LARGER than |dK21_full−host| (2.2e-3/1.9e-3/1.5e-3 vs ~4e-4), with
corr(full, kt) ≈ 0.43–0.58. Reading: the engine-map-on-plane-state
gap is LARGE, and the carried distinct memory CANCELS ~75–85% of it —
dK21_full is the residue of a screened pair (the same ~0.15 screening
constant found in the 08-30 kt calibration, now located structurally).
Consequences:
1. C4v2 as planned (accumulate leakage of known planes through bare
   transport) would reproduce the UNSCREENED map gap — wrong by 4–5×
   and half-decorrelated. Not run; superseded.
2. The closed form must produce the SCREENED propagator: the
   memory-feedback resummation that cancels most of each step's fresh
   leakage. This pairing is exactly what SOL's Jordan-quotient
   "intertwining and leakage gates" manage in the H4 case — his
   conditional pass reads, in hindsight, as the H4 instance of this
   screening structure.
3. REFINED TARGET (the lane's standing question): derive the
   screening kernel in closed form — the per-layer operator
   Σ = 1 − Memory∘MapGap⁻¹-style resummation — and its chain-motif
   matrix elements. The C1 adjoint grades any candidate in minutes;
   the measured screening ratio (0.75–0.85, layer-resolved, banked
   here) is the first thing any candidate must reproduce.
State: all executable branches measured; this derivation is the
lane's single open front. Best-matched worker: SOL (his formalism);
resumable by either from this entry.

## 2026-09-01 — Φ¹ out-and-back kernel VERIFIED exact in factored form

Small-n harness (`../public_g19_attribution_v1/c4v2_smalln_harness.py`,
n=32, dense reference): the k=1 screening kernel
Φ¹ = Π 𝒯 (1−ιΠ) 𝒯 ι  factorizes exactly into matmul chains —
three rank-n read families per lift pattern plus triple-diagonal
corrections; verified 7e-16 rel-max after fixing two factor-bookkeeping
bugs (the harness caught both — this is why the small-n gate exists).
Cost at n=1024: ~12 matmuls per source layer. NEXT: per-layer Φ¹ on
host planes at n=1024, summed over sources, tested against banked
dK21 (corr + C1-adjoint value share). The near-diagonal screening
measurement predicts Φ¹ carries the bulk; k=2,3 extensions use the
same now-verified slice-subtraction algebra if needed.

## 2026-09-01 — Φ¹ at n=1024: UNIT-β SIGNAL (the expansion is real)

Against the banked isolation target T_L = dK21_full − dK21_kt (pure
distinct-carry accumulation), the verified k=1 kernel scores corr
0.17–0.24 with UNFITTED β ≈ 0.95–1.22 at every layer L=5..14
(receipts `../public_g19_attribution_v1/C4V2_PHI1_rank0035.json`).
Unit amplitude with no fitting = the additive k=1 term of the
out-and-back expansion is being computed CORRECTLY; the remaining
target energy (‖T‖ ≈ 5–7×‖Φ¹‖) is deeper memory: sources m < L−2
propagating through (𝒯^QQ)^j with per-step plane-subtraction, plus
k≥2 excursions. All of it uses the same verified factored algebra
(transport families + lift-subtraction families, ~4 new families per
quotient step). NEXT: multi-step accumulation (sources back ~6–8
layers), same target test — expect corr to climb if the expansion
converges; then the C1-adjoint value grade, panel, and the deploy
assembly (billable plane-kt + Φ-sum ≈ engine-grade planes).

## 2026-09-01 — C4 gate verdict at k=1/depth-8 truncation: FAIL (honest)

Full measurement chain on banked r35+r100:
- Φ-sum vs its own target (full−kt): Frobenius corr 0.20→0.45
  (depth-8, monotone), unfitted β ≈ 1.1–1.3 — the expansion's algebra
  is RIGHT and converging in Frobenius.
- kt+Φ synthetic supply: r35 opt 16.6% but frozen 35→100 = −62.7%
  (kt's map-gap residue, still ~3× the true content at corr 0.45,
  dominates transfer) — inherits the kt disease.
- Φ vs (full−host): consistent NEGATIVE corr −0.18…−0.40 both nets
  (the screening seen directly; cross-net stable sign/scale).
- PURE host+Φ supply through the certified chain: capture 0.2%/0.1%,
  frozen transfer ~0 — the k=1 excursion content carries essentially
  NO consumption value despite its Frobenius alignment.
Preregistered C4 (≥8% both nets, unfitted): FAIL at this truncation.
Diagnosis: value-convergence of the expansion is far slower than
Frobenius-convergence; the value-bearing terms are the underived
ones (k≥2 excursions, completion-difference sources, t≥1 dressing).
The verified factored algebra + harness + all receipts remain the
foundation for whoever derives those terms.
STANDING CONTINUATIONS: (i) k≥2/birth-difference derivation
(weeks-class, value-convergence unproven), (ii) SOL's H4-exact
common-scalar chains (bypasses truncation entirely where the
property holds), (iii) large-truth-set exact numerics.

## 2026-09-01 — TERMINAL MEASUREMENT: the 5.8 : −4.8 : 1 cancellation

Value decomposition of the chain through the C1 adjoint (r35, banked;
receipts `../public_g19_attribution_v1/valdecomp.py` output):
  full-supply value      = 21.5  (the certified ~15.7% capture)
  map-gap (kt) component = +124.0  (5.8× the total)
  out-and-back component = −102.5  (−4.8× the total)
opposing layer-by-layer (e.g. L9: +34.2 vs −28.3). The consumed value
is the residue of a screened pair; assembling it from the two sides
requires ±3.5% value-accuracy on EACH, and the k-expansion measured
~0 value-corr at k=1/d8. VERDICT: the term-by-term expansion program
is CLOSED BY MEASUREMENT. The constant 5.8:−4.8:1 is the sharpest
input to any successor formulation: viability requires a
reorganization that tracks the residue directly (connected/
renormalized bookkeeping — the screening must be diagonalized in the
formulation, never assembled numerically). That is precisely the
ambition of SOL's Jordan-quotient program; hand him this number.
LANE STATUS: derivation branches closed (randomized reads, k-expansion);
open only as the residue-basis reformulation question + the H4-exact
common-scalar chains. All machinery (harness, factored algebra, C1
grader, this decomposition script) banked and reusable.

## 2026-09-01 — Step-level residue instruments: the complete characterization

New engine gates (HOST_SEED, KT_FROM_ENGINE) + banked runs give the
EXACT per-step identity (idcheck 0.0 at every layer):
  dK21_{l+1} = T1 (E-transport of old delta) + T2 (map gap on host
  planes) + T3 (quotient read R[Q_l]).
Measured (r35, receipts residue_analysis.py output):
- G_eff = T2+T3 generates 85% of the total value (15.3 of 17.9);
  transport T1 is minor (2.6). The value is PER-STEP generated.
- The step-level pair cancels 5:1 in norm and ~0.92 in value
  (−v_T3/v_T2 ≈ 0.86–1.05 per layer): the accumulated 5.8:−4.8:1
  constant is the per-step screening compounded.
- T2 is memory-free and exactly closed-formable. T3 is the memory
  object: the k=1 kernel is a correct additive part (per-step corr
  0.17–0.24, β≈1) and the d8 quotient model reaches Frobenius corr
  0.35–0.45 of T3 — but its VALUE share is −0.33 (wrong sign,
  +37 vs −113 through the C1 adjoint). The value of the quotient
  read lives in deep-memory components the transport-leak expansion
  cannot reach at any measured level.
FINAL STRUCTURAL STATEMENT: a viable supply must produce the
per-step screened generator G_eff (value ~1–5 units/layer) directly
as an analytic object — never as the difference of the ±20–30-unit
pair. All instruments to grade a candidate (per-step targets, C1
value metric, identity checks) are banked. The open mathematics is
the Jordan-quotient residue derivation (SOL's Q1–Q6 + this target).

## 2026-09-01 — Q-frame constants surfaced + leakage-measurement continuation map

Archaeology of the q-frame (V22 readout + QHEAD nodes) surfaced the
campaign's strongest capacity fact: **start5 exact-q oracle raw =
4.699e-10** (20× below goal; local-q residual scale ~4.65e-9), with
every LEARNED q-provider terminal (direct-transpose worse than
source-zero; product86's 86-feature/155k-param student recovered
0.094% of the parent-to-exact-q gap vs the frozen 8% gate). The exact
fused NCQC stream is the only q-provider class left — gated by the
correlated-ReLU leakage bounds (allowances 0.994%/1.39% per layer
D14/D10; generic witness 2.49%; diffuse bound ~1.13%).
NEXT-BLOCK CONTINUATION (the missing link): identify the CONCRETE
NCQC core matrices M_i (r=45..55) for the D11 card to measure the
actual spectral diffuseness χ and gate-coupling δ. Candidate homes,
in order: public_q24_qhead_full688_refit_v1 (feature_generator +
RESULT bindings), public_oracle_kappa_probe_v1 (ORACLE_DEEP /
FINAL_SENSITIVITY), LANE_matrix_polynomial_closure.md, and the
K33 DERIVATION.md coefficient conventions (17 derivative addresses,
q=38 table) which pin the Mehler-side normalizations.

## 2026-09-01 — Highest-value attribution question RESOLVED (+ the missing premise)

The DERIVATION_LEDGER's steering inequality hinged on whether the
recorded 0.78 was source alignment q_s or endpoint correlation q_v.
MEASURED (all 8 panel nets, banked rvecs): endpoint q_v = corr(r_v,
resid) = 0.28–0.45, mean **0.354**. Therefore 0.78 = SOURCE alignment
q_s, and the hopeful branch applies: success needs q_s 0.78→0.806
with exact cubic+quartic sources — NOT 0.943 endpoint correlation.
SECOND finding: the co-adaptation theorem predicts q_v≈0.97 from
q_s=0.78; reality gives 0.354. The theorem's premise (orders through
two absorbed by co-adaptation) requires the JOINT RETRAIN, which the
bolt-on LOO chain never had. Consistent with the month's five
co-adaptation failures. CONSEQUENCE: the minimum stack is exactly the
ledger's §8 — exact sources (NCQC fused stream) AND Gram-residualized
joint refit — both mandatory, neither sufficient alone. The refit
machinery exists (public_full688 infrastructure); the exact-source
build remains gated on leakage/fusion (Q1–Q6) with the M_i
identification as the next concrete step.

## 2026-09-01 — REST identified: the 12.7× m-feedback amplification

Perfect-state ladder replay with ALL marginals exact (m included),
card-class GC mean map, both nets: final-mean MSE **7.34e-10 (r35) /
7.28e-10 (r100)** — the same class as the exact-q oracle (4.70e-10)
and likely at/near the 8M truth-noise floor. Therefore:
1. The per-step consumption map is essentially exact; the ladder's
   9.25e-9 endpoint = per-step error amplified ~12.7× by the
   free-running m-feedback loop (the ladder arms fixed v/k3/k4 but
   never m). The "capacity floor" was an artifact of the arm design.
2. The REST question (SOL's REST_IDENTIFIABILITY lane) resolves:
   ~nothing is missing high-order information; ~everything is the
   m-channel accumulation structure.
3. GC-series upgrade attempts confirm the truncation is at a
   cancellation optimum (adding g3·g4 alone worsens 7.34→9.17e-10) —
   per the closure-partial-fixes law; no cheap map upgrade exists or
   is needed.
4. SUPPLY PRECISION SPEC (new, quantitative): any m-channel repair
   delivering accumulated error < 1e-8 needs per-step-equivalent
   precision ~1e-8/12.7 ≈ 8e-10 — i.e., per-unit mean-state repair
   at the few-1e-6 rms level per layer. The steering-inequality
   coefficients should be re-derived with the m-channel explicit.

## 2026-09-01 — THE M-CHANNEL: re-priced ladder puts m+v exact BELOW 1e-8

Extended SOL's perfect-state ceiling (10 nets, LOO betas, honest
frame) with mean-channel arms (receipts PERFECT_STATE_CEILING_M.json,
run_perfect_ceiling_m.py):
  m alone:      1.697e-8   (47% of the whole gap by itself)
  v alone:      1.466e-8   (the old first rung)
  **m+v:        7.483e-9   — BELOW the 1e-8 goal with two channels**
  m+v+k3:      6.859e-9
  m+v+k3+k4:   6.839e-9   (replaces the old 9.25e-9 "capacity floor")
Consistent with the 12.7x amplification finding: the m-channel is the
free-running accumulator no ladder arm ever fixed. CONSEQUENCES:
1. The steering simplex must be re-derived over (m,v) with k3/k4 as
   small crossing terms; required combined m+v deficit recovery for
   <1e-8: ~89.7% (from 3.197e-8, deficit 2.449e-8).
2. The m-channel is VIRGIN: no supply, student, or chain arm ever
   targeted per-layer per-unit mean error. Its propagation is the
   simplest of all channels (linear transport + tangent gains).
3. Next measurements: (a) per-layer m-error norms and structure
   (banked: host_m vs mc_pre_m); (b) cross-net stability of its
   principal structure; (c) which fraction the EXISTING engine
   supply's untapped content explains; (d) the m-channel's own
   irreducibility test battery (the same seven probes that closed
   the plane families - they have never been run on Δm).

## 2026-09-01 — M-channel RESOLVED: exactly derivative; the final synthesis

Corrected-index measurements (both nets, receipts in-session):
1. Δm_{l+1} = W_{l+1}ᵀ(cdf_l ∘ Δm_l) + fresh_l with transport explaining
   75–88%/step at UNFITTED β≈1.00 — and fresh_l is **100.0% explained
   (β=1.00)** by [state-error dds push + map residual], with the
   state-driver correlation alone 0.96–0.997. IDENTITY-GRADE closure:
   the mean error is an exactly-known linear integral of the
   {Δv,Δk3,Δk4} history through deploy-known maps.
2. Consequences: (a) the m-arm's 47% "virgin capacity" = historical
   state errors, not new information — the m+v<1e-8 landmark restates
   the ingredient problem, not a new channel; (b) the certified
   chain's transported-dds construction IS the exact m-recursion —
   consumption was never the bottleneck; the LOO betas ≪1 price the
   SUPPLY's mis-scale; endpoint q_v=0.354 is pure supply fidelity.
3. FINAL SYNTHESIS OF THE CAMPAIGN'S STRUCTURE: final error =
   exact-linear-integral of per-layer state errors; state errors'
   valuable content is identity-specific, unsummarizable,
   unlearnable, and screened (5.8:−4.8:1); consumption and transport
   are solved; the ENTIRE remaining problem is the exact analytic
   production of per-layer state errors at ~90% fidelity — the
   NCQC/Jordan derivation program (49-GEMM budget, leakage gates,
   SOL's Q1–Q6) with the small-algebra generic route already closed
   (correlation ceiling 0.47 at ≤49 channels) and the
   commutator/cyclic-module escapes as the live mathematics.

## 2026-09-01 — NCQC identifiability: empirical rank signatures (positive)

Stacked pair-moment matrix [[C,K21],[K21ᵀ,K22]] (block-normalized,
truth C + engine planes, banked): effective rank r90 34–67, r95
84–143, r99 339–454 (r35/r100 × L8/L12; deeper = more compact). Far
below full (2048) and below the r=45–55 NCQC working band's
~1000-class pair dimension — the energy-dominant joint pair structure
is consistent with a COMPACT quadratic-chaos core (r ≈ 15–30 by
r(r+1)/2 counting). Supports well-posed core identification
(NCQC_IDENTIFIABILITY lane) at lower rank than assumed. CAVEAT
(standing law): these are Frobenius-energy ranks; the value-bearing
content lives in the tail everywhere else we measured — the
identification should therefore fit cores on the energy-dominant
structure but be GRADED through the C1 value adjoint before any
conclusions about supply fidelity.

## 2026-09-01 — NCQC core fit: structure PASS, value FAIL (11th law confirmation)

Rank-24 core on 256 units (C+K21, 20% held-out entries, banked engine
planes, r35 L12): held-out C corr 0.9955, held-out K21 corr 0.8883 —
the plane content IS quadratic-chaos-generated; identification is
well-posed (receipts run_ncqc_corefit.py + ncqc_corefit_*.npz).
VALUE GATE: the fitted delta (fit − host) carries value −19.2× the
engine delta's (wrong sign; Frobenius delta-corr only 0.47) through
the C1 adjoint sub-block. The 11% energy misfit owns ~19× the value.
LAW (11th confirmation): energy-metric optimization is value-metric
garbage. NEW DIRECTIVE unique to this instance: the value metric's
OPERATOR structure (a2/e1 Mehler-weighted B-conjugations) is
deploy-known — only the residual projection γ is oracle. Next: refit
with the truth-free value-class weighting in the loss; grade against
the true adjoint. If the deploy-known weighting closes most of the
gap, compact cores revive as carriers.

## 2026-09-01 — Value-weighted core fit: sign fixed, +13% value share; the deployable form emerges

Truth-free value-class weighting (deploy-known a2/e1 operator
structure, γ:=1) in the fit loss: value share vs the engine delta
went **−19.2 → +0.13** (right sign; delta-corr 0.47→0.51). One crude
weighting (1+9·W), rank 24, 256 units — entirely unoptimized. The
energy-vs-value chasm is BRIDGEABLE by deploy-known weighting: first
mechanism all month to move value in the right direction by
construction choice.
DEPLOYABLE FORM (next): fit the core to HOST planes under exact
chaos-algebra consistency (C/K21/K22/K31 relations). The model's
rigidity then implies corrections to the host planes; test whether
(implied − host) aligns with true dK21 in Frobenius AND value. If
yes: a fully deploy-computable, no-oracle, no-learning* carrier
(*the fit is per-net self-consistency optimization on deploy-known
objects — cost = the optimization's flops, to be metered).

## 2026-09-01 — Rigidity-correction replication: NULL (ratio noise)

Grid (seeds 11/22/33 × r35/r100, RIGIDITY_GRID.json): value shares
scatter −5e291 … +1.1 … −4.3 — the +0.276 was a single-scalar fluke
on a near-zero sub-block denominator. The chaos-rigidity correction
at this scale/formulation has NO demonstrated value alignment.
Standing state of the core-fit thread: structure REAL (0.888
held-out K21 from rank-24 cores — the identifiability result holds),
value-weighting lever REAL at the fit level (−19.2 → +0.13 on the
engine-target fit), deployable rigidity value UNPROVEN (noise).
Full-plane fits with meaningful value denominators + the K22/K31
consistency constraints are the honest next scale for this thread —
multi-hour optimization work, to be preregistered before running.

## 2026-09-01 — Q2 test specification (executable next block)

QUESTION (SOL's Q2, the gate on the O(n)-Jordan loophole that beats
the 0.47-ceiling counting bound): does the quartic endpoint P4 factor
through the Jordan map J(A⊙B)=A∘B, i.e., does the value functional
annihilate ker J?

OBJECTS NOW AVAILABLE: fitted cores {a_i, M_i} (r=24, 256 units,
ncqc_corefit_rank0035_L12_r24.npz — held-out-validated 0.888). These
make the test concrete without the full-card identification:

PROTOCOL:
1. Kernel directions: for sampled unit pairs (i,j), compute the
   Jordan product M_i∘M_j and its kernel component: k_ij =
   U_{M_i,M_j} − (lift of M_i∘M_j) per the §1 polarization; the pure
   kernel witnesses are pairs with small ‖M_i∘M_j‖/(‖M_i‖‖M_j‖)
   (near-anticommuting pairs — measure their abundance in the fitted
   core first; if the fitted spectrum has none, Q2 is VACUOUSLY
   passed for this core class and the loophole is open).
2. Endpoint weights: the quartic channel of the CHAOS-frame endpoint
   per §9's external-label identity κ4(Q_i,Q_i,Q_j,Q_k) =
   48tr(M_i²M_jM_k) + 8tr([M_i,M_j][M_i,M_k]) — the SECOND
   (commutator) term is exactly the ker-J-visible part. Compute, for
   the fitted core and the C1-adjoint-derived endpoint weights
   (γ-free deploy version + true-γ grade), the fraction of endpoint
   κ4 value carried by the commutator terms vs the Jordan
   (first) terms.
3. GATES: commutator share ≤ 5% of κ4 endpoint value (both weight
   versions, both nets' cores, ≥3 fit seeds — the RIGIDITY_GRID
   lesson: single scalars are noise; require sign+scale stability)
   → Q2 PASS → the one-matrix Jordan stream is lawful and the
   49-GEMM build has a live route. Commutator share ≥ 20% → the
   quotient sees ker J → the O(n) loophole closes and the counting
   bound stands → the exact-source program needs r≥55-class
   noncommutative machinery (cost re-derivation required).
4. Cost note: all computations are r×r matrix algebra over ≤256
   units — minutes. The fit-seed replication is the expensive part
   (~15 min/seed on MPS).

## 2026-09-01 — DERIVATION: factored-adjoint commutator sums are budget-free

Q2 first datum (fitted core, 2966 sampled triples): commutator energy
share of κ4 = **0.246**, Jordan-term corr with total 0.973, and NO
near-anticommuting pairs exist (Jordan-ratio min 0.123) — the ker-J
sector is substantial; the pure one-matrix Jordan quotient would
discard ~25% of quartic structure. This forces carrying the
commutator sector — and the counting bound seems to forbid it. It
does not. DERIVATION:

**Claim.** With endpoint weights of factored (dyad) form
w_ijk = Σ_s α_i^(s) β_j^(s) γ_k^(s) of rank R, every commutator
endpoint sum
  S = Σ_ijk w_ijk tr([M_i,M_j][M_i,M_k])
evaluates in O(R·n·r³) work with NO carried H4 state.

**Proof sketch.** For one dyad (α,β,γ): define B = Σ_j β_j M_j,
Γ = Σ_k γ_k M_k (r×r accumulations, n·r² each). Then
  S_dyad = Σ_i α_i tr([M_i,B][M_i,Γ])
        = Σ_i α_i [ tr(M_iBM_iΓ) + tr(M_iΓM_iB) − tr(M_i²(BΓ+ΓB)) ]
— per unit: a handful of r×r products, ~4r³ MACs. At r=24, n=1024:
≈ 5.7e7 MACs per dyad per layer ≈ 0.03 GEMM-equivalents. The C1
adjoint rank measurement (r95 median ≈ 45–150 per layer) gives
R ≈ 50–150 ⇒ **1.5–4.5 GEMM-eq/layer** for the ENTIRE commutator
endpoint — inside the 49 budget for a 10–14-layer stream. The cubic
sums tr(M_iM_jM_k) reduce identically (3 accumulations + r³) and are
cheaper still.

**Why the counting bound doesn't apply**: the Ω(n)-separator argument
bounds CARRYING general H4 state through dense W. The factored form
never carries H4 state — the M_i bank (n·r² ≈ 6e5 floats) is the only
state, its per-layer evolution under the common-scalar tangent is
M'_a = Σ_i W_ia c_i M_i + births ≈ n²r² ≈ 0.3 GEMM-eq/layer, and all
quartic/cubic content is evaluated at the endpoint through the
factored adjoint. The 0.47 correlation ceiling bounded operator-
CHANNEL counts; the dyad-factored evaluation is not an operator
channel in that census — it inherits the FULL rank of the C1 adjoint.

**The remaining blocker, now singular**: DEPLOY-COMPUTABLE CORE
IDENTIFICATION. The offline Adam fit (4000 iters, MPS) proves the
cores exist and are compact; deploy needs either a closed-form
moment-matching construction (spectral, from C and K21 — the
identifiability lane's open algebra) or a metered-cheap iterative
solve. Everything downstream of identification is now derived to be
within budget.

## 2026-09-01 — DERIVATION II: rotation-free structure-table identification

The deploy-identification blocker dissolves. Chaos model: z_i =
a_iᵀg + Q_{M_i}(g). Key observations:
1. **C is exactly the Gram of the coefficient vectors**: with the
   isometry ⟨s(M),s(N)⟩ = ½tr(MN) and ξ_i = (a_i, s(M_i)),
   C = ΞΞᵀ. So Ξ = U√Λ from ONE eigendecomposition (0.17 GEMM-eq),
   determined up to O ∈ O(d), d = r + r(r+1)/2.
2. **K21 is LINEAR in the Jordan structure table given Ξ**:
   K21_ij = 2⟨ξ_j, ĝ(ξ_i,ξ_i)⟩ where ĝ = the Jordan multiplication
   table in the Ξ-frame. The rotation O never needs to be found:
   solve ĝ by least squares from (Ξ, K21).
3. **K22/K31 are linear in (ĝ⊗ĝ, λ⊗λ) given Ξ** — the Lie table λ
   (the commutator sector!) identifies from the card-carried
   fourth-order planes the same way.
4. **All endpoint sums evaluate through (Ξ, ĝ, λ)**: tr(M_iM_jM_k),
   the κ4 Jordan terms, and the commutator cross-Grams are table
   contractions of the ξ's — combined with Derivation I's factored-
   adjoint reduction, the ENTIRE exact-source endpoint is a chain of
   eigh + linear solves + r³-class contractions. No optimization at
   deploy, no rotation recovery, no H4 state.
Feasibility regime: per-ρ solve has d(d+1)/2 unknowns vs n equations:
determined at r ≤ 8 (d=44) from K21 slices alone; r = 16 (d=136)
needs the cum3-symmetry constraints (which triple the equation
count) or ridge regularization — to be validated numerically NOW on
banked planes against the fitted-core ground truth.

## 2026-09-01 — Derivation II validated at energy; value alignment open

Numerical validation (banked host planes, r35): the rotation-free
structure-table solve (eigh + one ridge LS) reconstructs held-out
K21 rows at 0.731 (r=8) → **0.919 (r=16) → 0.923 (r=24, plateau)**.
Identification is PROVEN: closed-form, deploy-computable, cheap.
VALUE GATE at full-plane scale (5 layers × true adjoint): implied
delta's aggregate value share 1.91 (right sign, 2× overscale) but
per-layer unstable (10× overshoots, one sign flip; Frobenius corr
~0) — per the RIGIDITY_GRID lesson this is NOT yet a supply; the
aggregate is likely cancellation luck. NEXT (exact steps): (1)
value-operator-weighted table solve (the −19.2→+0.13 lever, now at
full scale); (2) λ (Lie table) from K22/K31 joint constraints; (3)
per-layer consistency + r100 replication as the belief gate.
TONIGHT'S TWO STANDING DERIVATION RESULTS: (I) factored-adjoint
commutator sums are budget-free (1.5–4.5 GEMM-eq/layer at measured
adjoint rank — the counting bound does not apply to endpoint
evaluation); (II) rotation-free identification via structure tables
(energy-proven 0.92). The exact-source program now has a concrete,
fully-costed pipeline skeleton with ONE open alignment question.

## 2026-09-01 — Value-weighted table rigidity: r100 NULL (share −1.10)

Full-scale value-weighted solve (separable rank-1 deploy weighting):
r35 share +1.81, r100 share **−1.10** — wrong sign on the second net.
Twelfth transfer-law confirmation: static-rigidity deltas of
identified tables are not a supply in ANY tested formulation
(unweighted/-weighted, sub-block/full, both nets).
ROLE CLARIFICATION: the tables are the EVALUATION FRAME, not a
corrector. The deployable source they enable, with zero fitting:
**the chaos-exact marginal mean.** For z_i = a_iᵀg + Q_{M_i}(g), the
exact E[ReLU(z_i)] is computable from the spectrum of M_i (1D
integral over the quadratic-chaos law), while the card consumes only
the 4-cumulant GC series. The difference is a per-unit,
tables-derived, fit-free mean correction — the chaos-frame's
exact-consumption upgrade. NEXT UNIT: derive the spectral form of
E[ReLU(a·g + Q_M(g))], evaluate per unit from the identified tables
(needs per-unit M reconstruction OR a table-only route via moments
k5/k6 implied by ĝ — note the GC series at 4 cumulants is at a
cancellation optimum, so the chaos model's implied k5/k6, which ARE
table-computable without rotation, may be exactly the missing
even/odd tail), and grade per-unit mean-map error vs truth on banked
states. If the chaos-implied k5/k6 shrink the 7.3e-10-class per-step
floor OR (more importantly) the card-state mean-map error, the
12.7×-amplification lever engages with a fit-free deployable term.

## 2026-09-01 — Chaos k5/k6 completion: NULL as fit-free (tails unconstrained)

Cores fit on (C,K21) only → implied κ5/κ6 (closed forms verified
κp=(p−1)!·[½tr(M^p)+(p/2)aᵀM^{p−2}a]) OVERSHOOT (κ6 rms 3× κ3) and
worsen the mean map 4.5–6× on both nets: the fit pins spectra only
through 3rd order; higher traces are unconstrained. The completion
requires K22/K31 (4th-order) constraints in the identification —
the same augmentation the Lie-table solve needs anyway. NEXT UNIT:
joint (C,K21,K22,K31) core fit → re-test κ5/κ6 completion + the Lie
table in one pass. (Pattern note: the frontier is advancing by
verified elimination — every closed form and every null is banked;
thirteen falsifications now fence the space around whatever
survives.)

## 2026-09-01 — Chaos tail completion: CLOSED (state-driven, not tail-driven)

With K22/K31 constraints the implied κ5/κ6 collapse to sane scale
(8e-4/5e-4) and the mean-map completion is value-NEUTRAL (−1.3%/
−3.6%, both nets): the card-state map error is driven by STATE errors
(as the fresh-Δm identity proved), not by distribution tails. All
marginal-map improvements are bounded ~nil. Fourteenth closure.

**CONVERGENCE STATEMENT.** Every angle traversed tonight lands at
the same center: the pipeline (identification II + endpoint sums I +
exact consumption) is complete EXCEPT its source, and the source
must be the TABLE DYNAMICS — the causal recurrence for (Ξ, ĝ, λ)
under the layer map, whose discrepancy from the card's plane
recursion IS the state-error field. That is precisely SOL's
LIE_JORDAN / self_consistent_chaos_hierarchy lane ("explicit causal
recurrences for antisymmetric H3 commutators and symmetric
H4/variance squares"). The derivation program's entire remaining
content is that one recurrence, now with: the evaluation frame
built, the budget theorem proven, fourteen fences around every
shortcut, and instant grading for any candidate recurrence.

## 2026-09-01 — RECURRENCE DERIVATION, cell 1: the tables don't evolve — only Ξ does

Setup: z_i = a_iᵀg + Q_{M_i}(g), one layer z'_c = Σ_i W_ic φ(z_i).
Project φ(z_i) onto the degree-≤2 chaos of the SHARED latent g:
- linear coefficient: E[g·φ(z_i)] = E[φ'(z_i)]a_i + M_i·E[φ''(z_i)]a_i
  + M_i²·(next Hermite term)·a_i + … = f(M_i; unit stats)·a_i — a
  SPECTRAL FUNCTION of the unit's own core applied to its a.
- quadratic coefficient: ½E[(ggᵀ−I)·φ(z_i)] = c1_i·M_i +
  c2_i·a_ia_iᵀ + c3_i·M_i² + higher M-words, with per-unit scalar
  coefficients c_k = Hermite-type expectations of φ-derivatives under
  the unit's own (generalized-χ²) law — computable from carried
  marginal state + core spectrum.
KEY STRUCTURAL FACT: the latent g is SHARED across the layer map, so
the algebra tables (ĝ, λ) are FIXED. The entire dynamics is
  **Ξ_{l+1} = W_{l+1}ᵀ · F_l(Ξ_l)**,
with F per-unit built from ĝ-words (the Jordan square M² first):
the recurrence closes on the coefficient bank alone (n·d state,
matmul-class cost — matching Derivation I's budget). The Jordan
square inside F is how the flow explores the generated algebra
(SOL's cyclic-module setting). The card's plane closure is the
PLANE-PROJECTION of this same flow; the state-error field — the
singular remaining source — is their difference, now expressible
inside one formalism.
REMAINING CELLS: (2) closed-form spectral coefficients c_k(m,v,λ);
(3) ≥3rd-order chaos leakage/births (H3/H4 sector) and re-entry;
(4) small-n exact-MC validation harness (n=32, r=8): simulate, run
the Ξ-recurrence, compare planes — verify-first as with Φ¹.

## 2026-09-01 — Cells 4+: recurrence VALIDATED (structure exact, depth-stable)

Tiny exact world (n=32, r=8, MC ground truth):
- One-step word model: post-ReLU quadratic chaos coefficient = 
  {M, aaᵀ, M², I} words at per-unit R² 0.998–0.9998; linear channel
  {a, Ma, M²a} at 0.9985+; mixing identity machine-exact.
- Depth-8 iteration with DEPLOY-COMPUTABLE Gaussian-approx
  coefficients (c1=Φ(α), c2=pdf/sd): mean tracked at 5–8% rel
  (Gaussian closure: 37–44%), C at 23–48% (vs 78–94%), K21 at
  44–69%, all STABLE over depth — the closed table dynamics
  Ξ' = Wᵀ·F(Ξ) functions and does not blow up.
- The fidelity gap is LOCALIZED: the one-step degree-2 projection
  drops the ReLU's ≥3rd-order chaos leakage (L1: C 26%, K21 53%) —
  exactly the H3/H4 birth sector of SOL's hierarchy. Cell 3 = wire
  his birth derivations (normal-ordered squares, signed cubics) as
  additional table sectors; the constructive recurrence and his
  architecture have converged on the same object from both ends.
STATE: the fused-stream dynamics now EXISTS as validated running
code in its exact world, with cost structure matching Derivation I
and the remaining fidelity attributable to enumerated,
already-derived birth terms. Next cells: (2) exact spectral c_k
(replace Gaussian approx), (3) birth-sector wiring, (5) tiny-world
end-to-end vs the CARD-CLASS closure baseline (planes+4 cumulants)
— the honest comparison the Gaussian baseline understates.

## 2026-09-01 — Cell 3a: H3 birth sector closes on the word basis

Empirical third-chaos coefficient of φ(z) in the exact tiny world:
the 8–9-word basis in (a,M) — {a³, Sym(a,M), Sym(Ma,M), Sym(a,M²),
Sym(Ma,a,a), Sym(Ma,Ma,a), M-chain words} — fits per-unit at
**R² 0.951–0.992, median 0.979** (deeper M³ words hold the ~2%
tail). The H3 sector is table-expressible: emission = word-function
of (a,M) with per-unit scalars; transport = linear W-mixing.
HIERARCHY STATUS: H2 recurrence validated exact+stable (cells 1,4);
H3 basis closed (3a). Remaining cells: 3b (wire the carried-T
feedback into the depth iteration; fidelity target C/K21 ≤15% at
depth 8), 2 (exact spectral coefficients replacing the Gaussian
approx), 5 (card-class baseline comparison), then the 1024 bridge
via Derivation II identification. The fused stream is assembling
from validated parts.

## 2026-09-01 — Cell 3b first wiring: UNSTABLE (coefficient error, not structure)

(a,M,T) depth iteration with hand-derived feedback coefficients:
early layers improve on v1 (C L1 0.199 vs 0.266) but diverge from L6
(C 8.7, K21 5.5 at depth 8). Diagnosis: closure instability from a
wrong T-sector coefficient — the h3 = E[φ‴]-class scalar's sign was
flagged unresolved during derivation and a flipped feedback becomes
amplification. This is standard closure development, with the
harness-native fix: fit the TRUE per-unit T-emission/feedback
scalars from one-step projections (as 3a did for the basis), diff
against the analytic forms, correct, re-run. v1 (a,M) remains the
stable validated baseline (mean 5–8% at depth 8). NEXT: cell 2/3b
joint — exact coefficient calibration, then the depth gate again.

## 2026-09-01 — Cell 2 defined by calibration: couplings are SPECTRAL functionals

Coupling-classification over the (m,v)-Hermite family (synthetic
known-(a,M,T) states, mean-varied): only c(a) and c(M) are
Gaussian-family (R² 0.97/0.98, ≈cdf); all other sector couplings
(c(Ma), c(M²a), c(aa), c(M²), c(SymTa), c(T:I)) classify poorly
(R² 0.06–0.28) — they are spectral functionals of the unit's
generalized-χ² law, not (m,v) functions. Also measured: c(T:I) ≈
h2-class small (my 3·h2 was ~60× over — instability source #1
confirmed).
CELL 2 SPEC (the exact-coefficient derivation): all couplings =
E[φ^{(k)}(z_i)]-class expectations under z_i = m + aᵀg + Q_M(g),
whose characteristic function is closed-form
(Π_μ(1−2iλ_μ t)^{-1/2}·exp[imt − t²aᵀ(I−2iλ)⁻¹a/2]); ONE 1D CF
inversion per unit yields P(z>0), p(0), p'(0), p''(0) — every
coupling scalar — at ~1e3–1e4 flops/unit (free at deploy). Pipeline:
implement CF-inversion couplings → recalibrate against the harness
fits (gate: ratio 1±10% per coupling) → re-run 3b depth iteration →
the ≤15% C/K21 gate. The recurrence development is now a fully
specified, verify-gated engineering path.

## 2026-09-01 — Cell 2 first increment: CF-ladder linear couplings VALIDATED

Mode-resolved exact emission for the linear channel via the
characteristic function (closed-form CF of m + aᵀg + Q_M; Gil-Pelaez
P(z>0), p(0), p'(0) by 1D quadrature) with a two-level Stein ladder:
against empirical per-(unit,mode) projections, **corr median 0.9975,
β median 0.935** across units. The exact spectral-coefficient layer
works; the ~6% amplitude deficit is the truncated third ladder level
(mechanical to add). REMAINING SEQUENCE (all gated, no open
questions): third ladder level (β→1±3%) → quadratic-channel CF
couplings (same machinery, mode-pair resolved) → 3b-v3 depth gate
(≤15% C/K21) → card-baseline → 1024 bridge → panel. The exact
fused-stream recurrence is materializing coefficient layer by
coefficient layer, verified at every step.

## 2026-09-01 — THE COUPLING ENGINE: leave-one-mode-out exact integrals

The asymptotic Stein ladder diverges at level 4 (β med 0.864, range
0.48–1.17 — the GC lesson at coupling level). The exact engine
replaces it: per (unit, mode), z = c_μ(h) + z_rest with c_μ(h) =
ã_μh + λ_μ(h²−1)/2; F_rest = the rest-distribution-smoothed ReLU
(rest CF → FFT density → cumulative integrals, once per unit-mode);
couplings = Gauss-Hermite integrals of Hermite-weighted F_rest(c_μ).
VALIDATION: linear channel corr median **0.9998**, β median
**0.9885** (spread 0.81–1.18 on two units = FFT grid/tail artifacts,
tunable). No series, no divergence; generalizes the validated K3S
tracker's conditional-mean machinery. The SAME F_rest yields the
diagonal quadratic couplings via (h²−1)-weights; cross-mode entries
via leave-two-out 2D-GH where needed (words absorb most cross
structure). SEQUENCE: quadratic-channel engine → 3b-v3 with exact
couplings → depth gate ≤15% → card baseline → 1024 bridge.

## 2026-09-01 — 3b-v3 first assembly: mean-channel implementation bug (isolated to L1)

Quadratic engine v2 validated (corr med 0.998, β med 1.017, all
units, robust p0 off-diagonal). Full exact-engine depth iteration
then DIVERGES via the MEAN channel (μ_rel 0.60 at L1 → 63 at L8)
while C/K21 start reasonable (0.20/0.38 at L1). Since the L1 state
is exact, a 60% L1 mean error is an IMPLEMENTATION BUG in the
emit() mean extraction (the E[φ] path), not model error — the
validated channels (linear 0.9998, quadratic 0.998) are unaffected.
NEXT (single isolated check): emit()'s Emean vs empirical E[φ] per
unit at layer 1; fix; re-run the depth gate. Assets standing: both
exact emission engines, the depth harness, the v1 stable baseline.

## 2026-08-31 — MEAN-CHANNEL BUG FOUND: factor-2 in the CF (spec-level, certified fix)

Isolation: emit()'s per-unit Emean vs empirical E[phi] at the exact
L1 state = ratios 0.9379-0.9922 (1-6% LOW, worsening by unit). The
mixed mean is CANCELLATION-DOMINATED (E[phi]~0.5 through zero-mean
columns): idiosyncratic per-unit errors don't cancel — the 12.7x
amplification physics in miniature. v1-Gaussian survived because its
cruder errors are SHARED across units and cancel in the mix.
ROOT CAUSE (spec-level, in the Cell-2 entry's formula): for
z = m + a.g + Q_M(g) with Q_M = (g'Mg - trM)/2, the CF denominator is
**(1 - i t lam)**, not (1 - 2 i t lam) — E[e^{isg^2}] = (1-2is)^{-1/2}
at s = t*lam/2. The recorded CF also missed the -it*tr(M)/2 centering.
Correct CF: e^{itm} * Prod (1-it*lam)^{-1/2}
            * exp(-it*sum(lam)/2 - t^2/2 * sum(at^2/(1-it*lam))).
CERTIFICATION (direct test, n=32 r=8, N=2e6): buggy CF mean err ~70%;
corrected CF via E[relu] = (m+E|z|)/2, E|z| = (2/pi)int(1-Re CF)/t^2:
med 0.18% / max 0.44%; residual ~= the analytic tail (2/pi)/T — added
as exact correction in the rebuilt harness.
REBUILD: the ephemeral heredoc harness is now a FILE:
exact_recurrence_v3.py (corrected CF everywhere; direct-CF mean
channel; leave-one-mode linear + v2 quadratic engines; --gaussian arm
= v1 baseline in the SAME world for the honest in-world gate).
Running: gauss arm then exact arm, seed 9, depth 8, N=4e6.
Gate unchanged: C_rel & K21_rel <= 15% at L8.

## 2026-08-31 — 3b-v3 with corrected CF: mean channel FIXED; gate fails on TRUNCATION (structural)

Rebuilt-world depth-8, seed 9 (exact_recurrence_v3.py; in-world
gauss baseline). L1 sanity = MC noise (model formulas exact).
  gauss: mu 4-7%, C 35->71%, K21 72->93%
  exact: mu L2 **0.08%** (divergence GONE; 80x better than gauss
         one-step), L8 14.7%; C 16.4->53.9%; K21 37.2->74.3%
VERDICTS: (1) the CF factor-2 fix certifies — the exact mean channel
now survives the cancellation-dominated mix. (2) Exact couplings
halve pair errors vs Gaussian at every depth. (3) The <=15% gate
FAILS for (m,a,M) state REGARDLESS of engine quality: the L2 row is
the DEGREE-2 TRUNCATION FLOOR — one exact push loses 16% C / 37% K21
to the ReLU's >=H3 chaos, which the state cannot carry. Not engine
error (engines validated 0.998-0.9998). ECHO: one-step K21 loss 0.372
~ the 1024 card's measured supply q_v=0.354 — the tiny world
reproduces the campaign constant; the card's plane closure IS this
truncation.
NEXT (structural, before any H3 engine work): the CHAOS-ORDER
CEILING LADDER — sampled-emission recurrence, each layer's phi
empirically projected to H<=k of the shared g (exact couplings at
all orders by construction), k=2/3/4 in one pass on shared draws.
k=2 must reproduce today's exact arm (cross-check); if k=3 closes
<=15% at L8, the analytic H3 completion (3a words 0.979) has a
measured ceiling and proceeds; if k=4 fails too, the shared-latent
representation itself is the wrong object (redirect, cheaply known).

## 2026-08-31 — H3-completed readout DERIVED (pre-computed while ceiling runs)

If k=3 gates, the moment readout for state (m,a,M,T) is closed-form.
Wick parity enumeration of E[z_i~^2 z_j~] with cubic part C3:
existing three terms + FIVE new families, all bilinear/quadratic in
T against (a,M): 2<T_j, a_i (x) M_i>, 2<T_i, a_i (x) M_j>,
2<T_i, M_i (x) a_j>, <T_i T_j M_i>-chain, <T_i (x) T_i, M_j>.
C gains the H3 Gram: C_ij += (1/6)<T_i,T_j>. ALL are table
contractions -> the Derivation-II rotation-free frame extends to the
H3 sector (the identification aliasing note: card-carried C already
CONTAINS the H3 Gram additively — the joint identification must
split C = Xi2 Xi2' + G3 with G3 psd; K21's five new families give
the constraints). Budget: <T_i,T_j> Gram at 1024 = one
1024x816x1024-class GEMM ~ 0.8 GEMM-eq if T carried dense (r=16);
word-compressed (3a: 8-9 words) it is a 9x9 word-Gram combination.
BRANCH B (if even k=4 fails at depth): representation is complete as
k->inf (no fresh latent in the real net); the question is RATE; the
compressed-carry cell = "word-basis closure under mixing" (mix, refit
words next layer, R^2) — measurable in the same harness.

## 2026-08-31 — CEILING LADDER (seed 9): k=4 PASSES the gate; errors saturate, not compound

run_chaos_ceiling.py: sampled-emission recurrence, phi empirically
projected to H<=k each layer (exact couplings at every carried
order), shared draws. Depth-8 L8:
  k=2: C 30.4%  K21 53.7%  (mu 6.1%)
  k=3: C 18.8%  K21 26.5%  (mu 4.6%)
  k=4: C  6.6%  K21 12.3%  (mu 4.0%)   <- GATE PASSED (<=15%)
Facts: (1) truncation error SATURATES with depth at every order (L4
~= L8) — the recurrence self-stabilizes; per-step losses do not
compound. (2) ~2x decay per chaos order (kink's heavy tail). (3) The
analytic k=2 engines (L8 53.9/74.3) sit ABOVE their k=2 ceiling
(30.4/53.7): reprojection-each-layer beats accumulated engine error
— engine-quality gap exists but the ORDER extension dominates.
(4) The required chaos order is FOUR — the card's cumulant depth
recurring as chaos order.
DECISION: the fused-stream state = chaos coefficients through H4.
Feasibility fork for 1024 (r~16-24): dense T readout ~0.8 GEMM-eq
(affordable); dense H4 Grams ~68 GEMM-eq (NOT). The word-compression
question is now decisive -> cell W: does the CARRIED T/H4 stay
word-expressible in (a,M) under mixing (birth was 0.979)? Measuring
on saved k=4 coefficients (seed 10 replication running with --save).

## 2026-08-31 — Seed-10 replication + WORD-CLOSURE NULL (fifteenth fence)

Ceiling seed 10 (harder world): k=2 C52/K76, k=3 C34/K49, k=4
C16.2/K20.1 at L8 — ordering and ~2x-per-order decay REPLICATE;
absolute level is world-dependent (k=4 passes the 15% gate on seed 9,
marginal-fail on seed 10). Saturation replicates (L4~L8).
WORD-CLOSURE (run_word_closure.py, exact carried coefficients):
carried T on own-(a,M) words R^2 med 0.12-0.23; Q4 med 0.05-0.08;
H3/H4 energy shares ~13% each (saturating). MIXING DESTROYS
WORD-SPARSITY (birth 0.979 -> 0.12 after one W): the carried H3/H4
sectors are independent state, NOT unit-local functions of (a,M).
Fifteenth fence: the ~20-scalar/unit compressed 1024 carry is dead.
CONSEQUENCE: with order-4 necessary, no unit-local compression, and
dense H4 unaffordable at 1024, the O(n) Jordan-module escape (Q2)
is now THE decisive gate on the 49-GEMM build. Running the Q2
protocol next; per tonight's factor-2 lesson, first verifying the
kappa4 two-term identity (48tr(Mi^2MjMk) + 8tr([Mi,Mj][Mi,Mk]))
numerically under the 1/2-normalization before using it.

## 2026-08-31 — Q2 CLOSED BY DERIVATION: cumulant endpoints are Jordan-Grams

Two convention bugs fixed en route (same disease as the CF factor-2):
the recorded kappa4 identity's 48/8 coefficients are the UNHALVED
convention; under Q = (g'Mg-trM)/2 the exact form is
3tr(Mi^2MjMk) + (1/2)tr([Mi,Mj][Mi,Mk]) (CGF-derived, MC-checked).
THEN the derivation: polarization gives tr((A∘B)(C∘D)) = (T1+T2)/2
over cyclic trace classes, and the three pairings sum EXACTLY:
  **kappa4(qA,qB,qC,qD) = <J_AB,J_CD> + <J_AC,J_BD> + <J_AD,J_BC>**
  **kappa3(qA,qB,qC)   = <A, B∘C>**            (J_XY = X∘Y)
(machine-exact 4e-17; MC 1.5% at N=6e6). The "commutator term" was a
BASIS ARTIFACT: symmetric cumulant reads never see ker J. Q2 = PASS
analytically for the H2 sector — the one-matrix Jordan stream is
lawful for ALL kappa<=4 endpoint reads; the statistical fitted-core
protocol is superseded (its role shrinks to the H3/H4-sector reads).
OPEN residue: whether the H3/H4-sector reads (<T_i, M-inserted T_j>
contractions) and the RECURRENCE respect the quotient.
NEXT: the fifteenth fence killed UNIT-LOCAL words only. Shared-basis
carry T_i = sum_b theta_ib E_b is EXACTLY mixing-closed (mixing acts
on theta). Measuring the carried sectors' shared rank now (SVD of
per-layer H3/H4 coefficient blocks from the saved k=4 coefs).

## 2026-08-31 — Carried H3/H4 CONCENTRATE on a shared basis (deepening with depth)

SVD of the carried coefficient blocks across units (seed-10 saves):
T (dim 120): r90 15 -> 10 -> 7 -> 6 over preL2/4/6/8; Q4 (dim 330):
r90 16 -> 11 -> 8 -> 5. The recurrence funnels the higher sectors
into a SMALL SHARED subspace, and concentration INCREASES with depth
(the deep collapse). Shared-basis carry T_i = sum_b theta_ib E_b is
exactly mixing-closed; basis growth = birth directions only; the
deployable form is a streaming subspace tracker (accumulate birth
span, truncate to top-R) — GEMM-class at R ~ tens.
ARCHITECTURE NOW ASSEMBLED (every piece proven/validated/measurable):
state (m, a, M) + rank-R shared H3/H4 (theta per unit); exact linear
transport; per-unit CF engines (single-matrix, Jordan-lawful) + birth
projection; readout = theta-Grams through R x R tables + the five
derived K21 families; endpoint = Jordan-Gram identities (Q2 proven).
RUNNING: run_shared_rank.py seed 10 (hard world) — rank-R truncated
recurrence, R in {4,8,16} vs k2/k4 — the representation-ceiling gate
for the deployable carry (>=90% of the k4-over-k2 gain at L8).
CAVEATS for the 1024 bridge: energy-rank != value-rank (r-emission
lesson); n=32 bounds rank; generic vs net-shared cores cuts both ways.

## 2026-08-31 — Shared-rank arms (hard world L8): R=n/2 carries K21 fully; early breadth sets the rank

  k2 C52/K76 | k4 C16.2/K20.1 | R=4 C43/K56 | R=8 C35/K42 |
  R=16 C24.8/K19.3 (K21 = exact-arm level; C at 77% of the gain)
The preregistered R~8-12 gate FAILS: late-depth concentration (r90~6)
does NOT mean small-R suffices — errors made at the BROAD early
layers (carried rank ~15) persist through the saturating recurrence.
R = n/2 is the honest carry scale in the generic world. At 1024 that
is R~512: STILL thin-GEMM affordable (theta-mix ~R/1024 GEMM-eq;
readout Grams ~2R/1024) IF the basis stays factored through rank-one
births; the R^2 table builds are the tight item (~s GEMM-eq at
a-subspace rank s). Adaptive per-layer R (broad early, narrow late)
is the natural refinement.
PIVOT (shortest evidence-first): tiny-world structure questions are
answered. Next evidence must be 1024-real-net: through layer 2 the
chaos frame is EXACT and IDENTIFICATION-FREE (a_i = W1 rows; ReLU
1D Hermite coefficients; rank-one births; arccos-kernel closed
forms). Stage A = grade chaos-frame C (pre-L3, exact to k-truncation)
and K21 probe slices vs the card's carried planes vs banked MC truth
(mps_cov2 mc_pre_cov / host_cov) on rank35. If the chaos supply beats
the card at shallow depth on the REAL net, the mechanism transfers;
the deep extension (Derivation II identification + streaming tracker)
is then the build.

## 2026-08-31 — STAGE A at 1024 (rank35): frame VALIDATED; shallow card not discriminable

run_chaos1024_shallow.py (attribution_v1). Orientation fixed
empirically: z0 = W0^T x (col corr 0.99993), push z = W^T phi (0.81
vs 0.03). ReLU Hermite coefs exact (0.39894, 0.5, 0.28209, 0, ...).
Results (N=4e5, shared-draw arms k=2/4/12 through TWO nonlinearities):
- My k12 chain's C is INDISTINGUISHABLE from banked MC truth within
  the sampling floor (3.3% vs floor ~3.2%; preL2 2.8% vs 2.8%) —
  the identification-free chaos frame at 1024 is CORRECT end-to-end.
- Truncation ladder at 1024 (noise-cancelled): K21 loss k2 16-17%,
  k4 7-9% — tiny-world ordering replicates on the real net.
- Card: C near-exact shallow (0.6-0.7%). Its K21 CANNOT be judged by
  full-matrix Frobenius at this N: half-split noise floor (0.84/0.62)
  exceeds card-vs-chain (0.46/0.33) — null instrument, lesson =
  K21 discrimination needs diag+consumed-direction metrics or 10x N.
  Transpose artifact ruled out (K21T 1.33 >> 0.46).
Shallow depth is a no-contest region (card fine there; its failure is
mid-depth q_v=0.354). NEXT CELL (engine-at-1024): per-unit CF engines
on ~64 probe units of z1 — top-R modes by |lambda|+|a~| + Gaussian-
lumped tail (exact by CLT; IS the deploy structure) — exact marginals
+ couplings vs the shared-draw sampled truth. Tests the corrected
engines at REAL spectra; the last verify-first gate before the
streaming-tracker build.

## 2026-08-31 — ENGINE-AT-1024 CERTIFIED (rank35): all gates green at real spectra

run_engine1024_probe.py (attribution_v1): 64 probe units of z1's
H<=2 state, full 1024-mode corrected CF (log-space, principal branch
safe), leave-one-mode F_rest engines on top-8 modes, shared-rng
sampled truth N=4e5:
  Emean ratio med **0.99995** [0.9941, 1.00515]  <- the channel that
      diverged before the CF fix is now exact at 1024
  v ratio 0.99943 | k4 ratio 0.986
  linear couplings corr 0.99949 beta 0.98859  (gate >=0.995, 1+/-3%)
  quad couplings   corr 0.99782 beta 0.98363
  k3 ratio med 0.771 RESOLVED as reference noise: median |k3|/MCnoise
  = 1.79 (units' k3 ~0.01, floor ~0.006) — not an engine miss.
STRUCTURAL BONUS: ReLU's odd He coefficients >=3 vanish, so H3 has
ZERO birth at mean-zero units — z1's state is (m,a,M,Q4), no cubic;
H3 first births deeper where means are nonzero.
The engine layer is certified at real-net spectra. Replication on
rank100 running. REMAINING BUILD (the only uncertified piece): the
streaming shared-basis tracker carrying (m,a,M)+rank-R H3/H4 past
L2, whose ceiling the tiny world priced at R~n/2 with saturating
errors. beta deficits (-1.1%/-1.6%) echo tiny-world quadrature
truncation — systematic, calibratable.

## 2026-08-31 — r100 replication: engine certification holds on the second net

rank100, same parameter-free engines: Emean med 1.0003 [0.9756,
1.008] (one-unit -2.4% tail, wider than r35 — note for the extreme-
spectra units), v 0.99978, lin 0.99948/0.98813, quad 0.99788/
0.97861, k3 0.81 (same noise-dominated reference), k4 0.954.
Frozen-transfer satisfied: no fitting anywhere, both nets agree,
same ~1-2% systematic quadrature beta deficit (calibratable).
ENGINE LAYER: CERTIFIED ON BOTH NETS.

## 2026-08-31 — W1: fixed small-V entry FAILS (L2 state is broad); frame must be value-weighted

Out-of-subspace energy of the EXACT factored L2 state (rank35),
V = top-s eigvecs of normalized a-Gram + M-energy operator:
  s=64: a 70% / M 96% | s=128: a 50% / M 88% | s=256: a 25% / M 66%
  s=384: a 12% / M 42%
The entry layers are broad (matches tiny-world early r90 ~ n/2);
the spectral collapse is DEEP. A fixed energy-frame V from L2 cannot
carry the state. STANDING FACTS FOR THE TRACKER: (i) L2 is exactly
factored ANYWAY (dictionary {a_hat_i (x) a_hat_i}, per-unit weights)
— no truncation needed at entry; (ii) banked C1 adjoint r95 ~
105-163 at L3-14: the VALUE frame is ~150-dim at every depth — the
truncation metric must be the adjoint/value metric, not Frobenius;
(iii) readout Grams of the dense-factored form are GEMM-class
(<M_c,M_d> = u_c' (rho∘rho) u_d) but tr(M^2 M) chains cost ~n
GEMM-eq dense — the value-frame compression is where deploy cost is
won. NEXT: W1b — residuals of the L2/L3 state against the
ADJOINT-derived frame (value-weighted V), using the validated C1
machinery.

## 2026-08-31 — Recalibration + SIXTEENTH FENCE: Gaussian 3-point k3-push dead

RECALIBRATION (banked depth profile + map-gap reread): the card's C
is 0.6-1.4% Frobenius at ALL depths (diag <=floor 0.05% shallow,
~0.26% real deep; off-diag real ~1.2% deep; truth floor P=8M). Yet
exact-v alone buys 3.2e-8 -> 1.47e-8: the 1024 game is TENTHS OF A
PERCENT on deep v — perturbative-around-Gaussian regime (real-net
unit k3 ~ 0.01). Per-step needles (MAP_GAP reread): v-push 0.05-
0.26%, k3-push 6-11% (its "exactmap" arm was that script's own bad
approximation — ignore). Full-matrix K21 Frobenius at N=4e5 is a
null instrument (noise 0.62-0.84).
SIXTEENTH FENCE (run_k3push_hermite.py): the full Gaussian-Hermite
3-point k3-push (per-unit alpha_k from card marginals, rho from card
C, all factorized classes + exact (1,1,1) triangle on 96 probes)
lands 72-92% error vs the card's 6-11% — 11x WORSE, both shallow and
deep. ANCHORS PROVE THE FRAME on the same objects: full-pair v-push
0.74% (corr 0.9992), mean-push 0.07%. VERDICT: next-layer k3 is
carried by the STATE'S OWN non-Gaussianity (k3/K21 transport), not
by Gaussian mixing structure — the state-driven law at push level.
The card's k3-push deficit is therefore either (i) H3-state beyond
its K21 planes or (ii) truncated consumption of carried planes.
NEXT: OE-attribution — the engine's truth-plane exposure gates
(oe/oeall) split (i) vs (ii) directly: if truth-K21 injection leaves
the 6-11% k3-push error, consumption-truncation dominates (fit-free
richer series fixes it); if it removes it, supply dominates (the
tracker's job). Banked oeall variants may already answer this.

## 2026-08-31 — THE V-CHANNEL REFRAME: supervised residual field at 1e-5-grade truth

Value ledger recount: k3-channel total worth ~2-8% MSE (ladder
7.48->6.86; chain split k3=2.2pts). Exact v alone = -54%; exact m+v
= 7.483e-9 BEATS THE GOAL. The entire game is the per-step v-push:
card cumulative v-error 0.05->0.26% (per-step fresh ~0.02-0.05%,
compounding), identical both nets.
NEW INSTRUMENT FACT: the banked truth is a SHARED-DRAW system — push
identity diag(W^T mc_post_cov[l] W) vs mc_pre_v[l+1] = 1e-5 (not the
0.05% independent-noise floor). Corrections are gradable at
0.001%-level. W-convention nailed: z_{l+1} = W[l+1]^T phi (all
layers, both nets).
ARCHITECTURE VERDICT: do not replace the card's pair machinery
(Gaussian-3-point fence + card's 0.6-1.4% C prove it's excellent);
SUPPLY ITS RESIDUAL: R_l = mc_post_cov^truth - Cov_post^card = the
exact field the certified chain's dv-hat channel tries to supply
(measured q_v=0.354). Chaos-frame H3/H4-Gram pair completions are
the candidate generator, now aimed at a supervised target with
exact truth. NEXT BUILD: the R_l extractor — reconstruct the card's
post-cov from its carried state via its own emission formulas, bank
R_l per layer per net, then grade every supply candidate (chaos
Grams, engine terms) by value-weighted alignment against R_l.
FULL CIRCLE: the night's machinery (certified engines, H3-birth
structure, shared-basis economics, state-driven laws) now serves
raising q_v 0.354 -> 0.9 against a 1e-5-grade target.

## 2026-09-01 (post-midnight) — THE V-RESIDUAL INSTRUMENT ARC: form deficit isolated; three nulls; coadaptation at field level

Banked instruments (attribution_v1): vresid_field_rank{0035,0100}.npz
(R_l per layer, f32) + VRESID_FIELD jsons; gram_split files (noise-
walled, kept for provenance); run_vresid_field.py / run_gram_split.py
/ run_gram_verdict.py / run_k3push_hermite.py.
MEASUREMENTS (both nets unless noted):
1. R_l = truth-post-cov − engine-pair-closure(card state): 0.64→1.5%
   Frobenius, pushed diagonal ≈ the card's next-v error at EVERY
   layer (the field explains the card's v deficit completely).
2. dv-hat (certified chain) vs exact target: corr 0.03→0.82, beta
   0.002→0.78 with depth — deep decent, shallow-mid badly under-
   supplied; base and dep frames identical.
3. NULLS: so2/dM2 kappa3^2-terms (R unchanged 0.25% rel); Mehler
   tmax 60 (unchanged); Gaussian 3-point k3-push (72-92% vs card
   6-11% — state-carried non-Gaussianity, 16th fence). G-split via
   rho-coupling: noise-walled at N=2e5 (needs ~1e9; G5/G6 canaries).
4. Value-geometry: the Frobenius-dominant rank-one deep mode of R
   (share 34-61%, u ~ 0.7-corr with m and k3, replicates both nets)
   is VALUE-EMPTY in the pushed-v metric (removal leaves 82-100%);
   rank-8 leaves 50-84%. The v-value lives in the BROAD BULK.
5. FORM-VS-SUPPLY SURGERY (decisive): truth marginals+C into the
   closure halves Frobenius R but removes only 0-25% of pushed-v
   deep (L5 even worsens 1.21x): the v-value deficit is 75-100%
   CLOSURE-FORM error. Better inputs cannot fix the existing form.
6. COADAPTATION AT FIELD LEVEL (strong-world tiny test): exact
   H3/H4-Gram corrections OVERSHOOT the total form error many-fold
   (closure −2 to −14) — the closure's low-order errors currently
   CANCEL its H3/H4 deficit; partial Gram patches are net-negative
   (the closure-partial-fixes law at the pair field). Any completion
   must replace the pair model's chaos content WHOLESALE.
7. Perturbative-world ladder (SA .45 SM .04): k4 C 1.1→9.1% over
   depth — the tiny world cannot reach the 1024 regime (kappa3 20-30x
   too large, n=32 lacks width-CLT); regime-matching is a dead end.
NEXT 1024-NATIVE CELL: exact bivariate pair integral on probe pairs
(the H<=4-state 2D law of (z_a, z_b) via joint quadratic-chaos CF)
vs truth post-cov entries vs the closure — measures whether a
RICHER DEPLOYABLE CONSUMPTION SERIES (the expansion of that exact
integral) can close the form deficit, i.e. whether the build target
is "richer series on carried state" (cheap) vs "carry H3/H4 state"
(the tracker). run_pair_exact_probe.py.

## 2026-09-01 — FORK DECIDED: the pair deficit is 100% INFORMATION-LIMITED (19th fence)

run_pair_exact_probe.py (rank35, L5/8/11/13, 160 probe pairs each =
top-|R| + random): exact 2D Gram-Charlier quadrature over the SAME
carried information vs the closure's Mehler series:
  rms ratio exact/series 0.992-1.010, error corr 0.984-1.000.
The series HAS CONVERGED to the exact integral of its own inputs.
VERDICT: no richer consumption series on the carried state can close
any of the form deficit — the carried pair-state {m,v,C,k3,k4,K21,
K22,K31} simply lacks the information (H3/H4 cross-structure beyond
the planes). Nineteenth fence, and a POSITIVE one: it eliminates the
last competing build target.
THE BUILD IS NOW UNIQUELY DETERMINED (every alternative measured
dead): the 1024 fused-stream pair readout — grow the carried state
with the chaos H3/H4 sectors and produce the pair field WHOLESALE
(chaos-consistent all orders, per the coadaptation law; partial
patches measured net-negative). Targets and instruments banked:
R-fields both nets at 1e-5-grade truth; value metric = pushed-v;
form deficit 75-100% of v-value; W-conventions nailed; engines
certified; W1 says entry-layer state is broad (exact factored form
free through L2) and the value-frame at depth is ~150-dim (C1
adjoint r95). OPEN ENGINEERING QUESTION (the whole remaining
program): the depth-transport representation between L2-exact and
the ~150-dim deep value frame.

## 2026-09-01 — THE DELTA-FRAME LAW: the card cannot be rebuilt, only supplied

run_plane_recursion.py (standalone card-class plane recursion from
the engine's own emissions + exact pair-level pushes, validated
term-by-term against the map-gap formulas): accumulates v 3.7% /
k3 33% / C 5.0% by L14 — 14x WORSE than the card (0.26%/11%/1.4%).
One-step isolation at FULL-TRUTH inputs (truth planes from
mps_planes, now noted as banked): my pair-truncated K21 push errs
34% on k3 — algebra verified correct; the dropped generic triples
are simply large. THE CARD ACHIEVES 6% ANOTHER WAY: its co-adapted
correction machinery (collective-skewness/K3S/injection chain)
absorbs the generic-triple content statistically. Exact-shallow
hybrid arm (l0=3): helps ~14% relative — immaterial next to this.
LAW (completes the coadaptation law): the deployed card is a
CO-ADAPTED ARTIFACT — first-principles reconstruction of any of its
channels lands far below it; the ONLY integration path for new
supply is the DELTA FRAME: the card exactly as-is + residual
corrections graded against the banked R-fields (pushed-v metric),
integrated jointly (retune) per the coadaptation law. This is also
why the certified 18.7% chain was built as injections.
BUILD LEDGER (what the grown-state readout must produce, all now
measured): per-layer pair-field residual R_l (banked, 1e-5 truth);
target metric pushed-v; the supply must come from information beyond
{m,v,C,k3,k4,K21,K22,K31} (19th fence); wholesale-consistent chaos
orders (coadaptation); shallow layers identification-free (chaos
frame exact through L2, engines certified); deep value frame
~150-dim (C1 adjoint). Truth pre-planes K21/K22/K31 for l>=2 are
BANKED in mps_planes_rank*.npz — the supply-grading harness needs no
new sampling at plane level.

## 2026-09-01 — THE ORACLE DEPTH CURVE: the v-supply prize priced layer by layer

run_estimated_consumption with dv-hat layer-window oracle arms
(truth Δv replacing the chain's dv-hat on chosen layers; fixed
universal betas; both nets):
  only_v (r35/r100): L2-3: 0.1/4.0 | L2-4: 5.7/9.7 | L2-5: 2.1/11.3
  | L2-7: 7.6/14.7 | ALL: **46.9/53.7** (corrected raw 1.252e-8 /
  1.458e-8; total reduction 53.2%/46.9%)
The full v-channel oracle reproduces the ladder's exact-v ~-54%
INDEPENDENTLY — instrument coherent end-to-end. Marginal value
~4-6 pts/layer beyond L4 (Δv targets grow with depth).
STRATEGY NOW PRICED: (a) the shallow analytic supply L2-L4
(identification-free: L2 arccos-exact; L3 word-Gram pair field —
birth-words are pre-mixing so the word-closure NULL does not bind;
z2 has NO cubic sector, ReLU odd He coefs vanish; L4 one composed
step) ≈ +6-10% raw estimated — AT the meaningful-fire floor, and
co-fireable with the held f32 pack (-4.9% adj). (b) each deeper
exact layer ~+4-6 pts — the tracker's incremental payoff schedule.
(c) full v-channel = the 1.3-1.5e-8 raw regime; with kappa channels
-> the ladder's 7.5e-9. CAVEAT: estimated-consumption instrument
(fixed betas, estimated not board); the deployable realization
captures a fraction of each oracle number; board = judge. Oracle
variant files: dvhat_rank*_shorc*.npz.
NEXT BUILD (the deployable increment): dv-hat_chaos at L2-L4 —
run_chaos_supply_shallow.py producing the analytic Δv per net from
the identification-free frame + certified engines + word-Grams.

## 2026-09-01 — Stage-1 shallow supply: two numerics laws certified; the class SATURATES at 0.28%

run_chaos_supply_shallow.py (analytic v(L2) from the identification-
free frame). Debug ladder with certificates at every step:
1. Rectangle-rule CF-inversion bug (THIRD convention-class catch):
   the t->0 plateau overweight adds a constant dt/(2pi) density
   offset — at 1024-scale it inflated E[phi^2] 8x (excess mass
   0.112 and m2 +7.85 matched the prediction EXACTLY). LAW: all
   CF-inversion integrals use trapezoid endpoints. The probe's
   F_rest carries the same offset (re-certification running).
2. CF-only moment identities certified (no density anywhere):
   E[phi^2] = (E[z^2] - (2/pi) int Im[CF'']/t dt)/2 (0.02% on MC);
   p(0), p'(0), p''(0), P(z>0) as single Gil-Pelaez integrals.
3. Exact-variance completion (arccos all orders -> Gaussian
   remainder mode) + copula-shaped >=3-sector tail pinned to the
   exact per-unit residual.
RESULT LADDER: v2_rel 6.53 -> 0.031 (trapezoid) -> 0.0034 (variance
completion + tail) -> 0.0028 (CF-only scalars). Diag ratio 0.9974.
**SATURATION**: the (m,a,M)+Gaussian-remainder unit-law class
bottoms at ~0.28% on v(L2) vs the card's 0.051% — the residual is
STRUCTURAL (the H4 sector correlates with (a,M) through shared
modes; Gaussian-remainder mispricing + off-diag Gram truncation).
VERDICT: the shallow deployable (+6-10% oracle) is NOT reachable in
this machinery class — the card is too good shallow. Realization
requires either (a) Q4-extended unit laws + exact G3 Grams (bounded,
multi-block analytic machinery), or (b) the ANCHORED-DELTA DEEP
route: one-step chaos-complete pushes anchored on the card's own
carried state at l>=8 where the card's error grows to 0.26% and
anchoring kills accumulation — needs Derivation-II identification
of (a,M)-equivalents from the card's carried (C, K21). The deep
targets are 4-5x bigger and the oracle prices each deep layer at
+4-6 pts. NEXT BLOCK: (b) — the identified anchored one-step
supply, graded against the banked R-fields.

## 2026-09-01 — Trapezoid law closes the engine beta deficit (re-certification)

run_engine1024_probe.py with trapezoid-endpoint F_rest (rank35):
lin corr 0.99949 -> **0.99982**, lin beta 0.98859 -> **0.99938**;
quad beta 0.98363 -> **0.99717**. The "systematic quadrature
deficit" was entirely the rectangle-rule offset. The coupling
engine layer now stands at the 0.1-0.3% level, parameter-free.

## 2026-09-01 — Delta-model at order<=2: NULL (third confirmation of the fork)

myR = [chaos Gram readout G1+G2+tail] - [closure(card L1 state)]
graded against banked R_1: corr **-0.0005** (frob/pushed closures
negative). The order-<=2 chaos readout's deviation from the closure
is ORTHOGONAL to the true residual — R is the H3/H4-cross
information, confirmed now from three independent directions
(information-limit probe, coadaptation overshoot, delta-model null).
NOTHING below exact G3-sector Grams supplies it.
NEXT BLOCK (specified, verify-first): the G3-EXACT machinery —
per-unit third-chaos emission coefficients of phi(z) as words with
CF-exact scalars (the H3 BIRTH basis, valid pre-mixing at 0.979),
cross-unit word-Grams via trace algebra, assembled into
Delta-Covpost = G3-Gram - closure's implied 3rd-order content (M3
gating, already built). Tiny-world first (exact coefficients banked
in the ceiling harness saves), then 1024 anchored on card state at
deep layers via Derivation-II identification. Grade: corr/pushed-v
vs banked R-fields; then the oracle consumption for value. All
engines now 0.1%-certified; the three numerics laws stand guard.

## 2026-09-01 — G3-1 CERTIFIED + the supply architecture completes

run_g3_birth_words.py: the analytic H3-birth model (third Stein
ladder, words {a3, sym(a,a,Ma), sym(M,a), sym(M,Ma), sym(M2,a)},
CF-only scalars E[phi'']=p(0), E[phi''']=-p'(0), E[phi'''']=p''(0),
normalization E[d3 phi] = 3! T_dense):
  perturbative world (SA .45 SM .04 = the 1024 regime):
    L1 R^2 med **0.9989** (p10 0.994) beta 0.984  <- GATE PASSED
  strong world (seed 10): 0.61 — the ladder diverges at kappa3~0.4
  (GC-lesson; not the deploy regime). Solo-word diagnostic: sym(M,a)
  dominant (0.89-0.96), assembly beats every solo.
  L3/L5 birth-only: R^2 NEGATIVE — deep T-emission is dominated by
  TRANSPORT of carried T, as the word-closure NULL predicted.
SYNTHESIS (the G3-supply architecture, every piece certified
separately): T-Gram at depth l = sum over l'<l of per-layer analytic
birth-Grams (G3-1) propagated by E[phi']-weighted W-transport chains
— which IS the chain's validated tangent transport (run_tangent_v2's
D). Components: (i) per-layer (a,M) via Derivation-II identification
from carried C/K21 (energy 0.92); (ii) certified birth words; (iii)
the existing D-transport. The composition = the deployable G3-Gram
supply feeding Delta-Covpost, graded against banked R-fields.
NEXT (G3-3): verify the composition in miniature — transported-birth
accumulation vs the exact carried T (pre-coef order-3 rows) in the
perturbative tiny world, all pieces exact. Then the 1024 assembly.

## 2026-09-01 — G3 word-class CERTIFIED COMPLETE; exact-scalar engine specified

Diagnostics on the transport recursion (perturbative world):
emission L1 0.9989 -> preL2 0.9975 (mix exact) -> emission L2 0.67
-> collapse. Residual attribution at L2: Q4-consumption word 18.6%,
passthrough beta correct (+2.7% of pz), B∘T Wick term negligible.
THE DECIDING MEASUREMENT: word-class LS ceiling per unit with the
full dictionary {6 birth words, M-chain, 2 Q4-contractions
(a∘₁Q4, Ma∘₁Q4), T-passthrough, M∘₁T}:
  L2 R^2 med **0.9898** | L4 **0.9822** (p10 0.983/0.849)
The ELEVEN-WORD STRUCTURE SPANS THE DEEP EMISSION. Failure was
scalars-only (Stein-ladder truncation — diverges beyond leading
order even perturbatively for the transport-active words).
EXACT-SCALAR ENGINE (G3-2 spec, no ladder): E[∇³φ] EXACTLY =
E[φ''' d⊗d⊗d] + 3sym E[φ'' d⊗M]  (only two terms; ∂³z=0), with
  E[φ'' d_i] = p(0)·E[d_i | z=0]                (conditional mean)
  E[φ''' ddd] = −∂_w[p(w)·m3(w)]|₀              (conditional 3rd)
Conditional moments of the GAUSSIAN d = a+Mg given the quadratic z:
per-unit 1D machinery (tilted CF / F_rest class — the validated K3S
conditional-mean family). The words emerge as the m-functions'
structure; the exact scalars = values/derivatives at w=0. All
per-unit, deployable at 1024. Same treatment extends to the Q4 words
(E[∇⁴φ] two-term identity). NEXT: implement + certify per-unit
against the tiny-world exact emissions (targets: emission R^2 >=0.98
at L2-L6), then the transported-Gram assembly and the R-field grade.

## 2026-09-01 — G3 SUPPLY CHAIN VERIFIED END-TO-END IN MINIATURE (the q_v~0.9 level)

run_g3_exact_emission.py: the RESOLVENT-EXACT emission (no ladder):
  E[phi''' d3] = (1/pi) Re int it CF [m_t^3 + 3sym(m_t x C_t)] dt,
  m_t = (I-itM)^-1 a, C_t = M^2(I-itM)^-1 (tilted-Gaussian closed
  form per eigenmode); E[phi'' d] = (1/pi) Re int CF m_t dt.
  Emission gates: L1 R^2 0.9997 (beta 0.9998), L2 0.976, L4 0.921,
  L6 0.956 — the ladder's divergence eliminated exactly.
run_g3_transport.py with the exact engine: the transported-birth
recursion (passthrough pz*T + resolvent births + exact linear mix)
tracks the EXACT carried T at
  preL2 0.9997 | preL3 0.967 | preL4 0.928 | preL5 0.890 | preL6
  0.875 (beta 0.89-1.0) — error SATURATES (self-stabilization).
Per-unit T-hat ~0.9 => implied G3-Gram supply fidelity ~0.9 — THE
TARGET LEVEL (chain's measured q_v = 0.354). The architecture
(identify -> resolvent birth/emission -> D-transport -> Gram) is
now verified end-to-end in miniature with zero fitted parameters.
1024 counterparts: identification (Derivation II, 0.92 energy),
engines (0.1%-certified), transport (the chain's validated D).
NEXT: (G3-4) the Gram/DeltaC assembly + tiny-world R-closure value
test; then the 1024 lift (identified anchors, resolvent solves per
unit at top-modes + tail, Gram via t,s-double-integral contraction
tables). Deploy-cost engineering after fidelity.

## 2026-09-01 — G3-4 verdicts: supply machinery VERIFIED; patch-integration DEAD from five angles; route = dv-channel

run_g3_value_tiny.py (perturbative world, L4/L6):
1. **G3hat ≡ G3exact at the value level** (corr/frob/pushed metrics
   identical to 2-3%): the transported analytic supply reproduces
   the exact H3-Gram. THE SUPPLY MACHINERY IS FULLY VERIFIED.
2. G3−M3 patch: overshoots 4-6x (M3-gate subtracts the GC-marginal
   terms — wrong object). Surgical Mehler-t3 swap (G3 − copula
   rho^3-term): still overshoots 1.6-3x, corr ~0 — the closure's
   H3-content is SMEARED across its series x pattern web (its
   K21-driven dM terms model H3-overlap at other Mehler orders).
   NO decomposition isolates an insertion slot. Fifth angle closed.
INTEGRATION LAW (final): the pair closure admits NO patches of any
form. Lawful routes: (i) full replacement (measured worse in every
regime), (ii) joint re-derivation/retrain, (iii) **the dv-hat
consumption channel** — the certified chain's own route (18.7%):
supply Δv-estimates through the consumption machinery with LOO
betas; co-adaptation happens via the beta-fit; the supply needs only
CORRELATION with the true Δv (q_v). The G3 supply at 1024 therefore
targets dv-hat: Δv̂_G3 per layer from the identified state's
G3-Grams, graded against the banked Δv targets (tv−hv, both nets)
exactly as the chain's dv̂ was graded (corr 0.03→0.82; the
shallow-mid gap is the open territory).
TINY-WORLD PROGRAM COMPLETE. The 1024 lift = Derivation-II
identification per layer → resolvent engines (certified) →
transported G3-Grams (verified) → dv̂_G3 → the chain's consumption.

## 2026-09-01 — First 1024 G3-dv signal: NULL — and the DIFFERENCE-FIRST law completes

run_ncqc_corefit full-layer L5 (1024 units, r=24): held-out C 0.968 /
K21 0.929 — identification holds at width. run_g3_dv1024.py: the
identified cores' resolvent T-emission Gram, pushed to dv-hat (raw
and copula-subtracted): corr with true Delta-v = -0.035 / -0.001 vs
the chain's 0.586 at the same layer. NULL, and consistent with the
tiny-world fifth angle: R = (true H3 content) - (the card's SMEARED
H3 model); computing either side absolutely cannot produce the
difference (small difference of large objects; systematics do not
cancel across different model families).
**THE DIFFERENCE-FIRST LAW** (the integration picture, complete):
residual supplies must be computed AS differences of MATCHED models
— the certified chain's own architecture (dK21 = engine-with-tensor
minus engine-without, same family) is why its dv-hat reaches 0.59
while every absolute construction lands at 0. The G3 machinery's
1024 role is therefore INSIDE the chain's engine: upgrade the tensor
recursion's per-step emissions with the resolvent-exact forms and
take the chain's own with/without difference. The engine's k3u
already beats the card (3.6-6.8% vs 6-10.5%); its remaining error is
its own truncation — the resolvent emissions target exactly that.
NEXT BLOCK: engine-internal G3 upgrade (swap the engine's
emission_K21/GC-series per-step pushes for resolvent-exact versions
where they bind), regenerate the chain's dK21/dv-hat, grade the
delta against the banked targets, then the consumption oracle.

## 2026-09-01 — T-family K21 Wick coefficients CERTIFIED: (1, 1, 1, 1/2, 1)

MC coefficient solve (12 instances, r=5, N=1.5-8M) + full Wick
derivation agree: for z = alpha.g + Q_B(g) + C3_T(g)/6 (Wick-ordered),
  K21_ab = tr(B_a^2 B_b) + a_a'B_b a_a + 2 a_a'B_a a_b       [H2 base]
    + 1*<T_b, a_a (x) B_a> + 1*<T_a, a_a (x) B_b>
    + 1*<T_a, B_a (x) a_b> + (1/2)*<T_a (x) T_a, B_b>
    + 1*<T_a, (B_a (x) I (x) I) T_b>                          [chain]
(measured [1.05, 1.01, 1.09, 0.45, 0.85], LS residual 1.9% = MC
noise; parity closes all other families). This is the exact post-K21
assembly for the resolvent-emission state (alpha, B, T) — the
engine-delta candidate dP = resolvent-K21post(cores) −
engine-K21post(core-implied stats) is now fully specified with
certified pieces end to end: resolvent T (0.9997), Stein-word
alpha/B (0.998-0.9998 tiny; 0.1%-class at 1024), and these
coefficients. NEXT: implement run_g3_engine_delta.py, push dP,
grade against the chain's UNSUPPLIED residual (dv_true − beta·dv̂),
stackability through the consumption betas.

## 2026-09-01 — Engine-delta first signal: POSITIVE but weak (0.18/0.10); attribution running

run_g3_engine_delta.py (L5 cores, exact resolvent post-state
alpha/B/T — no ladder — + certified (1,1,1,1/2,1) assembly, minus
engine emission on core-implied stats): pushed-diag candidate
  corr vs dv_true 0.1835 | vs chain residual 0.1033
  (chain itself 0.5863; the absolute-Gram null was -0.03)
First POSITIVE 1024 G3 signal — the difference-first form works
directionally. BUT dP_rel = 1.26: the two "matched" sides disagree
at full scale — one is off (candidates: identification error cubed
through the T-terms; an assembly convention; the engine side's
implied-k4/K22 handling). Attribution rerun in flight comparing both
sides against the truth-fed post_5 instrument plane. The stacking
question (does 0.10-vs-residual survive refinement and transfer to
r100) is the block's open gate.

## 2026-09-01 — Engine-delta attribution: machinery CERTIFIED, identification is the bottleneck

Attribution of dP_rel=1.26: K21r (resolvent side) 114% off the
truth-fed reference with diag 3.4x undersized, K21e (engine side)
27.5% off, corr 0.972 — MY side broken at 1024. Then the skipped
certificate (the law, honored late): synthetic unit, post_state ->
assembly vs MC: alpha corr 1.0000 beta 0.9983 | B 0.99998/0.9986 |
T-diag 0.9987/0.9979 | k3post assembly ratio 1.076 (the 7.6% = the
known missing T^3/Q4 families). MACHINERY EXACT. Therefore the 1024
failure = the FITTED CORES: 7% held-out identification error
amplifies through the cubic assembly (B^3, T cubic-in-core) to 114%
at the plane level. THE IDENTIFICATION IS THE BOTTLENECK of the
whole G3 lift — value-rank fidelity, not machinery.
NEXT (the specified identification upgrade, from the Derivation-II
banked note): JOINT (C, K21, K22, K31) core fit (the cum3-symmetry
constraints triple the equations; K22/K31 pin the fourth-order
spectra) + multi-seed; gate = the engine-delta's K21r_vs_ref << 0.27
(beat the engine's own miss) and corr_vs_resid >> 0.10, then r100
transfer. First positive signal on the board: corr_vs_dvtrue 0.18 /
corr_vs_resid 0.10 with cores this bad — headroom is real.

## 2026-09-01 — Joint-fit iterations: the r=24 MODEL-CLASS WALL

Certified K22/K31 (a,M)-formulas (MC-solved integer coefficients,
0.5%/2.4% resid; pure-Q parts independently re-confirm Jordan-Gram):
  K22 = tr(Mi2Mj2) + 2tr((Mi∘Mj)^2) + 2aMj2a_i + 2aMi2a_j
        + 8 a_i(Mi∘Mj)a_j
  K31 = 3tr(Mi3Mj) + 6a_i(Mi∘Mj)a_i + 6a_iMi2a_j
Joint corefit attempts at L5 (r=24, 1024 units): 4th-order losses
DEGRADE the 2nd/3rd fits in every configuration tried (w=0.35 all
entries: C 0.61/K21 0.83; w=0.05 top-decile masked, smooth K21
target: C 0.754/K21 0.868 vs base 0.968/0.929). CONSISTENT with the
banked kappa5/6 lesson: an r=24 (a,M) core pinned by (C,K21) cannot
also satisfy the 4th-order planes — the MODEL CLASS is too small,
not the optimization. NEXT: r-sweep (48, 64) base-vs-joint; if the
wall persists, the identification-noise-robust assembly (shrink
T-terms toward the noise level) or the table-only Gram route.

## 2026-09-01 — r=48 joint fit: optimization collapse; the identification SWEEP specified

r=48 same-hyperparameter joint fit collapses (held-out K21 0.034,
loss 87 — untrained: lr/init/iters tuned for r=24 do not carry).
The identification wall now has two coupled axes: MODEL CLASS
(r=24 provably too small for joint 2nd-4th order) and OPTIMIZATION
(larger r needs its own hyperparameters). This is systematic sweep
work. THE SWEEP (SOL-shape, fully specified):
  grid: r in {32, 48, 64} x lr in {1e-3, 3e-3, 1e-2} x l4-weight in
  {0, 0.02, 0.05, 0.15} x iters 8000 (cosine decay) x seeds {11,12};
  layers L5 + L8; targets: C=mc_pre_cov, K21=tensor_planes pre
  (smooth), K22/K31=mps_planes truth top-decile masked;
  gates per cell: held-out C >= 0.96 AND K21 >= 0.92 AND (joint
  cells) 4th-order heldout corr reported; winner -> the engine-delta
  gate (run_g3_engine_delta.py <rank> <L> <corefile>): K21r_vs_ref
  << 0.27, corr_vs_resid >> 0.10; then r100 frozen transfer.
Everything downstream of the winner is built and certified
(resolvent post-state, Wick assemblies, difference-first candidate,
consumption oracle). The one open lever is core fidelity.

## 2026-09-01 — THE a/M SPLIT DEGENERACY: the identification bottleneck's true face

Per-piece certificates PASS on the failing real units (alpha 0.9999,
B 0.9999, T-diag 0.96/beta 1.008) yet the k3post assembly captures
only 5-32%. Cause: the fitted cores are M-DOMINATED (quadratic part
= 0.32 of 0.35 unit variance; |a| med 0.17) — for near-chi^2 units
relu's third moment lives in H4+ sectors beyond the H<=3 assembly.
But the REAL L5 state is a-dominant (relu alpha_1 + width-CLT). The
(C, K21) fit has a MASS-SPLIT DEGENERACY between aa'-Gram and
(1/2)MM-Gram and chose the unphysical branch; r-growth makes it
WORSE (more M-room). K22/K31 pin the split in principle (tr M^4
mass) — the l4-loss fights were the optimizer resisting the a<->M
mass move, not noise. FIX: a-heavy initialization (top-r C-factor
scaled to ~2/3 diag mass) + ANNEALED l4 (0 -> 0.05 across training).
Sweep axis updated accordingly. Scale/spike/m-regime hypotheses all
measured dead en route (implied-C scale 0.94/1.007; error broad;
m-sweep reproduces at m=0).

## 2026-09-01 — Physical-split gate: resolvent miss HALVED; the two-axis trade isolated

a-heavy annealed fit (r32): a-share 0.645 (physical), K21 held-out
0.926, C 0.899. Gate: K21r_vs_ref 1.08 -> **0.609** — the
assembly-regime hypothesis confirmed (the H<=3 assembly engages on
a-dominant cores). But K21e_vs_ref 0.145 -> 0.597 (the engine side
pays for the C drop): the two core-fidelity axes TRADE under the
annealed loss — (C,K21)-accuracy vs physical a/M-split. Persistent
signal floor: corr_vs_dvtrue 0.18 across all three core versions.
HYPOTHESIS IN TEST: the degeneracy means (C,K21)-loss is ~flat
across splits — the a-heavy INIT alone (l4w=0) should hold the
physical branch AND keep C~0.99: sweep_r32_ainit_w0 running. If it
lands (C>=0.98, K21>=0.93, a-share>=0.6): the both-axes cores go
back through the gate; targets K21r_vs_ref << 0.145-class and
corr_vs_resid >> 0.11 with BOTH sides accurate.

## 2026-09-01 — Init-only test: FLATNESS REFUTED — (C,K21) actively prefers the wrong branch

sweep_r32_ainit_w0: despite the a-heavy init, at l4w=0 the optimizer
MIGRATED to M-heavy (a-share 0.109) while reaching C 0.990/K21
0.9405. The (C,K21) objective is not flat across the split — K21's
dominant tr(Mi^2Mj) term pulls mass to M. THE PHYSICAL SPLIT IS ONLY
IDENTIFIABLE THROUGH 4TH-ORDER PRESSURE (the l4-anneal was right in
principle; its C-cost 0.899 is the current frontier). Cell in
flight: NIT=20000, anneal completing at 3/8 of training, C-loss
weight x3 — recover C under the full-l4 equilibrium. The identified
split axis is now a measured, steerable quantity: a-share {0.109
without l4, 0.645 with} — the sweep's decisive dimension.

## 2026-09-01 — PER-LAYER PARITY WITH THE CERTIFIED CHAIN (the arc's crest)

The "stubborn 0.18" decoded: the chain's OWN single-layer injection
windows grade corr-vs-total-dv_true at L6: w55 **0.1808**, w44
0.148, w33 0.03, w22 -0.01; 4-layer window w58 0.405; full chain
0.586 — per-layer supplies run ~0.15-0.18 and the TRANSPORT
ACCUMULATION builds the total. My G3 difference-first candidate at
L5: **0.1806** — EXACT PARITY with the certified chain's per-layer
supply, from closed forms with ZERO fitted parameters (vs the
chain's cum3-tensor machinery). Core-quality refinements moved the
plane errors (K21r 1.08->0.573 best) but the value signal was at
parity throughout — the per-layer supply level is the construction's
property, robust to core grade (matched-degradation).
THE REMAINING BUILD IS NOW THE CHAIN'S OWN SHAPE: (1) cores at
L2..L13 (the fit-cell per layer; sweep-shape), (2) accumulate the
per-layer G3 deltas through the tangent transport (run_tangent_v2's
D machinery — exists, validated), (3) the JOINT value test: does
dv_true ~ chain + accumulated-G3 beat chain alone (the corr_vs_resid
0.04-0.11 per layer is the orthogonal budget; x10 layers through
accumulation = potentially material), (4) consumption oracle ->
fire decision. Alternative cheap probe first: multi-layer stacking
at 3 layers (L4, L5, L8 cores) before the full 12.

## 2026-09-01 — FIRST MEASURED RAW VALUE: 3-layer G3 stack +1.1% through the real chain

land_stack.sh (L4/L5/L8 cores, per-layer deltas, gamma-augmented
dvhat, consumption): reduction 2.1% -> **3.2%** (+1.1 raw points,
in-sample gamma {0.032, 0.037, 0.085}). THE DEPTH TREND: candidate
corr_vs_dvtrue L4 0.157 / L5 0.181 / L8 **0.257** (resid-corr up to
0.156; K21r_vs_ref down to 0.392) — deeper layers have bigger
targets AND better identification (L8 cores 0.965/0.950, best yet —
the spectral collapse aids the fit). Extrapolation to 12 layers ~
+4-5% estimated (sublinear guard). CAVEATS: in-sample gamma (mirage
law -> r100 frozen transfer is the belief gate, launching);
estimated-consumption instrument (fixed betas), not the board.
The G3 program has crossed from mathematics to measured raw value on
the certified chain's own consumption path.

## 2026-09-01 — FROZEN TRANSFER PASSES: the G3 supply is REAL on both nets

land_r100.sh (r100 cores L5/L8 via the same recipe — quality
transfers: 0.928/0.919 and 0.967/0.954; deltas; r35-FROZEN gammas
{0.0374, 0.0852}; consumption):
  per-layer corr: L5 0.171 (r35 0.181), L8 0.157 (r35 0.257)
  plane fidelity: K21r_vs_ref 0.565/0.387 (r35: 0.573/0.392)
  consumption: -12.2% -> **-11.6%** = +0.6 raw points FROZEN
  (r35 in-sample +1.1; the r100 base's -12.2 is the kt-arms' known
  transfer problem, separable — the G3-stack's own delta is positive
  on BOTH nets).
THE MIRAGE LAW IS SATISFIED: a zero-fitted-parameter, closed-form,
per-layer supply at chain parity that FROZEN-TRANSFERS positive —
the first new raw mechanism to do so since the certified chain.
Notes: r35's L8 strength (0.257) did not fully transfer (0.157) —
cross-net variability to map; the gamma is crude (the consumption's
LOO-beta machinery is the proper integration). CONTINUATION: all
12 layers (cores L2-L13 both nets), LOO-beta integration, virgin
gate, then the board decision (co-fire with the f32 pack -4.9% adj).
Estimated all-layer frozen value ~+2-4% raw — below the fire bar
alone; the fire case rests on the stack + f32 + any further depth
strength. The supply mechanism itself: PROVEN.

## 2026-09-01 — THE VALUE PROFILE INVERTS: early layers pay; L11 value-dead; exact-shallow is the aim

Consumption solo map (r35): L4 +0.5 | L5 +0.3 | L8 +0.2 | L11 <= 0
(any weight, either sign — deep-v corrections cancel in the
mean-channel; the amplification integral favors EARLY corrections;
sum ~= joint +1.1, additive). The corr-vs-depth trend (0.157->0.257)
is VALUE-INVERTED through the consumption. The chain's own weakness
is also early (corr 0.03-0.5 at L2-L5) — the G3 supply's incremental
room concentrates EXACTLY where the state is IDENTIFICATION-FREE:
the exact shallow frame (L2/L3: a,M exact from weights, factored,
zero fit error). NEXT CELL: exact-shallow deltas — the resolvent
machinery on the exact L2/L3 cores (top-64 eigenmodes + certified
vres Gaussian tail per unit), then the stack re-price. This closes
the loop with the identification-free frame: its saturation was at
ABSOLUTE v-supply; as a DIFFERENCE-FIRST delta it plays at the
layers the consumption pays most.

## 2026-09-01 — L2 exact-shallow delta: NULL — the value curve is a HUMP

run_g3_exact_shallow_delta.py (identification-free L1-post cores,
zero fit error): corr vs dv_true(L2) = -0.056 (chain: 0.034). No
target exists at L2 — the card is nearly exact there (0.05%, its
floor). Combined with the solo map (L4 +0.5, L5 +0.3, L8 +0.2,
L11 ~0): THE VALUE CURVE IS A HUMP peaking at L4-L5 — early enough
for the m-integral amplification, late enough for a real target.
Priorities: fill L3/L6/L7 (in-hump), retune the weak L4 fit
(0.900/0.893, the hump peak deserves the best core), THEN the
LOO-BETA INTEGRATION (the fixed universal betas may misprice deep
corrections — the proper consumption refit including the G3 stack
could revive the deep layers and lift the whole package). Honest
running total: 3-layer +1.1 in-sample / +0.6 frozen; hump-filled
estimate ~+2-2.5 in-sample; fire case still needs the LOO-beta
multiplier and/or k3-channel stacking (the improved K21r planes also
carry k3-supply — unexploited).
