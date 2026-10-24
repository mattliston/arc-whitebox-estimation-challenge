# LANE — public_g20_emission_retrofit_v1 (2026-09-01, teammate A "emission-retrofit")

Question: does bringing the engine's plane emission / exact plane push INTO the deployed card
(public_d11_util_halving_v1/estimator_cap.py, D11 lineage of board #328622) cut raw MSE, at what
billed cost, and does the learned F2J correction survive? Frozen files untouched; all work in
copies here (estimator_retro.py, run_retro_screen.py), every change env-gated.

## 1. Term-level diff (engine run_tensor_recursion_v4.py vs card estimator_cap.py)

Plane accuracy measured first (r35, off-diagonal rel error of the PRE K21 plane vs 8M-sample MC
truth `mps_planes_rank0035.npz`; card plane = `host_cov_rank0035_base.npz` k21; engine = the
instrument-frame / deploy-frame `tensor_planes_rank0035{,_dep}.npz` pre_L):

| L | card rel (corr) | engine rel (corr) | deploy-frame engine |
| 2 | 0.0853 (0.9964) | 0.0811 (0.9967) | 0.0812 |
| 3 | 0.0892 | 0.0668 | 0.0670 |
| 5 | 0.0861 | 0.0550 | 0.0550 |
| 8 | 0.0936 | 0.0590 | 0.0583 |
| 12 | 0.1252 | 0.0803 | 0.0790 |
| 14 | 0.1330 | 0.0937 | 0.0920 |

The engine's plane is ~35% more accurate than the card's at every layer >= 3 (the parent's
"both ~7%" reading was the engine's closure_rel diagnostic, not the card's plane).

EMISSION (post-K21 from pre state):
- Engine (:59-120): Mehler series in rho to t=20 (columns to q=30); first-order Edgeworth
  channels dM = {k3_a/6 (3,0), k3_b/6 (0,3), K21/2 (2,1), K21^T/2 (1,2), k4/24 (4,0),(0,4),
  K22/4 (2,2), K31/6 (3,1), K31^T/6 (1,3)}; optional second-order dM2 (kappa3^2 class) is OFF
  (SECOND_ORDER unset) in every run that produced the +10% supply; connected
  K21 = M21 - 2 r1 M11 - r2 r1 + 2 r1^2 r1 with r1 = edg2_relu_mean (1+L3+L4+L3^2/2), r2 = Edgeworth raw2.
- Card: `_q24_context` (:464) Mehler powers rho^0..rho^6 only (_Q24_TERMS = 7);
  `_edgeworth_weights` (:540) = the SAME nine first-order channels PLUS seven L3^2 cross products
  (17 channels, `_four_pair_edgeworth`); `_connected_planes` (:634) = the same connected formula
  (raw2 = Edgeworth-corrected, :1458). Diagonal: raw moments (`_copy_with_diagonal`).
  => At the emission level the card is RICHER in channels and SHORTER in the rho series (7 vs 20).
  No engine input is missing in the card: dense pre_k21 (`_cp_read`, :1801), dense K22/K31
  (`_cp4_read_planes` + `_push_quartic`, :1823-1837), m, v, k3 = diag(pre_k21), k4.

CARRY (what happens to the post-K21 plane between layers) — the real difference:
- Engine: plane slices of the full tensor are overwritten with the emitted post-K21, distinct
  entries = g g g T_pre + completion, exact 3-mode push. Everything the emitted plane generates
  at later layers is kept.
- Card: residual R = k21_target - cp_post_k21 (`_reset_stars`, :1280) is stored as the star
  `cubic` (n x n f32), pushed EXACTLY for ONE layer as a plane-only tensor
  (`_push_cubic`, :1134-1156 = W2^T(diag W) + 2 (W o (R_off W))^T W + W2^T (R_off W)), then the
  star is reset from the next target. The distinct entries created by pushing a plane-only
  residual (sum_i w_i (x) w_i (x) (R W)_i, a rank-1024 CP object) are DROPPED at every layer.
  Their plane contribution two and more layers later is what the card lacks.
- The CP3 births themselves (p = gate o C[:,c], q = e_c, lambda = pdf/sigma) are exactly the
  order-2 path word; carried exactly; cap 16384 = full birth, non-binding for births alone.

## 2. Retrofit built: RETRO_RESID_CP=1 (estimator_retro.py, `_retro_append_residual`)
Carry R exactly inside the card's own CP3 machinery: per unit i one (p,p,q) term with
p_i = e_i, q_i = R_off[i,:] + (R_ii/3) e_i, lambda_i = 1, so T_iij = R_ij and T_iii = R_ii under
the card's triple-permutation convention (`_cp_k3`/`_cp_read`); the star is zeroed. From then on
the CP push/gate/read handle it exactly at every later layer (what the engine does). Rank grows
2048/layer; the cap (16384) binds at layer 8 → energy-ordered merge (lambda^2 |p|^2 |q|^2, same
law as `_cp_grow`). Certificates: (a) `cert_resid_cp.py` — CP read of the pushed appended terms
vs `_push_cubic` of the same R: rel 4.9e-16 (off+full), 5.9e-16 (diag); (b) in-situ RETRO_CERT=1
prints the same identity on the live card at the first retro layer.
Cost: no new GEMM class; the CP push/read arena is fixed-size (bill measured below).

## 3. Results (r35 / r100; noise-corrected L16 MSE; deployed baseline r35 2.6758e-8, r100 2.7464e-8)
Harness certificate: RETRO off reproduces the deployed output (F2J on: −0.0% both nets, max |Δ| 1.2e-5 / 1.6e-5 on L16; u_local 0.600 / 0.618; 69-75 s per net). In-situ certificate (RETRO_CERT): CP read of the appended slice vs `_push_cubic`(R): rel 3.9e-5 (L2) → 1.2e-4 (L7), f32 accumulation. Receipts: RETRO_SCREEN_*.json, screen_*.log.

NEW FACT: F2J OFF the host closure alone is 8.59e-8 (r35) / 7.80e-8 (r100) — the learned F2J operator removes 69% / 65% of the host's raw. Any host change shifts the residual F2J was fit on.

| arm (r35) | u_local | F2J on | F2J off (host) |
| deployed (RETRO off) | 0.600 | 2.676e-8 | 8.59e-8 |
| residual carried exactly (merge policy, arena-schedule cap) | 0.760 | 7.60e-8 (+184%) | 1.127e-7 (+31%) |
| same, scale 0.4 (rest in the one-step star) | 0.702 | 7.14e-8 (+167%) | 1.128e-7 (+31%) |
| same, scale 0.2 | 0.702 | 6.57e-8 (+145%) | 1.026e-7 (+19%) |
| r100: deployed | 0.618 | 2.746e-8 | 7.80e-8 |
| r100: residual carried exactly | 0.786 | 9.73e-8 (+254%) | 1.222e-7 (+57%) |

Reading: carrying the plane residual R exactly (which is what the engine's "overwrite the slices, then push the full tensor" does to ITS residual) makes the HOST worse on both nets, and the damage saturates already at scale 0.4 (0.4 ≈ 1.0) — a threshold, not a proportional response. Two candidate causes, separated by the runs queued below: (a) the arena-schedule merge drops low-energy CP BIRTH terms from layer 8 on (the deployed card never drops a birth; the cap ladder shows raw is hypersensitive to that: cap 8192 → +351%); (b) the distinct entries generated from a plane residual are wrong physics for the card's Q24 target (the star's one-step plane-only carry is the better model). Bill: the CP push/read bill IS rank-dependent (+27% u at 2×rank), so a residual carrier is not free either.

## 4. Plane accuracy of the retrofitted card (r35, PRE-K21 off-diagonal rel error vs 8M MC truth; PLANE_ACCURACY_residcp_r35.json)
Dump certificate: the RETRO-off card's dumped planes equal `host_cov_rank0035_base.npz` k21 exactly (rel 0.0, corr 1.0, all layers).
| L | card | residual carried exactly (merge policy) | engine (deploy frame) |
| 3 | 0.0892 | 0.0713 | 0.0670 |
| 4 | 0.0866 | 0.0654 | 0.0584 |
| 5 | 0.0861 | 0.0669 | 0.0550 |
| 6 | 0.0856 | 0.0708 | 0.0537 |
| 7 | 0.0913 | 0.0781 | 0.0553 |
| 8 | 0.0936 | 0.0836 | 0.0583 |
| 9 | 0.1002 | 0.1119 | 0.0629 |
| 11 | 0.1176 | 0.1638 | 0.0732 |
| 14 | 0.1330 | 0.1951 | 0.0920 |
The exact carry closes about half of the card→engine plane gap through layer 8 (the physics of the carried residual is right), and degrades from layer 9 on, exactly where the arena-schedule merge starts dropping terms (the deployed card never drops a birth). Meanwhile the HOST MSE rises monotonically with the carried fraction even at scale 0.1 (+9.8%, planes only ~10% improved): better planes, worse final mean. That is the cancellation/coadaptation signature (phase-1 law re-observed at 1024×16): the host's own error budget and the learned F2J payload are tuned to the card's plane-error pattern. Consequence: a plane-accuracy retrofit cannot be judged without a JOINT retune — at minimum an F2J refit on the retrofitted host (`public_deploy_composition_prep_v1/run_joint_refit.py`: frozen protocol, 200-net rows (35-234), 160/40 split; needs the retrofitted card run on 200 nets ≈ 200 × 80 s ≈ 4.5 h CPU, fit in minutes). The previous joint refit (composed card) came in FLAT vs base (3.1339e-8 vs 3.1334e-8 valid-40).

## 5. Bookkeeping and policy isolation (r35, host = F2J off)
| arm | host MSE | vs deployed host 8.593e-8 | F2J on |
| scale 0 (zero residual terms, same rank growth/merges) | 8.594e-8 | +0.0% (clean no-op) | 5.66e-8 (+112% vs 2.676e-8) |
| merge policy, scale 0.1 / 0.2 / 0.4 / 1.0 | 9.43e-8 / 1.026e-7 / 1.128e-7 / 1.127e-7 | +9.8 / +19 / +31 / +31% | — |
| protect-births policy, scale 1.0 / 0.4 | 1.619e-7 / 1.398e-7 | +88 / +63% | — |
Findings: (i) F2J-on is CONFOUNDED in every retrofit run: the native tape feeds `entering_cp_rank` to the learned operator, and the residual terms double it (scale-0 leaves the host bit-identical yet F2J-on jumps to 5.66e-8). Fixed (birth-only rank reported when RETRO is on); the F2J-on numbers above predate the fix. (ii) Dropping residual terms (protect) is worse than dropping weak births (merge): at depth the carried residual content matters more than low-energy births. (iii) With any exact residual carry the host mean worsens although the L3-8 planes improve — cancellation/coadaptation inside the host, independent of F2J.

## 6. The retrofit planes graded the way the engine's were: as a POST-HOC supply (tangent → consumption, r35)
`tensor_planes_rank0035_g20resid.npz` = the residual-carry (merge policy) card's planes; `run_tangent_v2.py` → `dvhat_rank0035_g20resid.npz` → `consumption_betas.py` v-arm only:
only_v = −3.8 / +0.5 / +1.7 / +2.3 / +2.5 / +2.2 % at β_v = 0.82 / 0.6 / 0.5 / 0.4 / 0.3 / 0.2 (engine `_dep`: +10.6 at 0.4).
Per-layer tangent dv_corr: +0.41, +0.39 at the early injection layers (engine ≈ 0.50 at L5), then −0.17, −0.38, −0.49 in the truncated deep half (planes worse than the card's from L9). The early-layer exact carry supplies real value; the arena-schedule truncation destroys it at depth. Deployable form implied: card UNCHANGED (host, tape, F2J intact) + a side CP carrier of the star residual + the tangent's linear response at frozen β — the same architecture the instrument validated at +10% for the engine — with the deep-layer truncation removed (bigger arena: rank ≤ 2× → memory 6.8 GB, bill +27% util as measured) or the carrier limited to a depth window (chain3/4 arms).

## 7. FIRST HOST GAIN: residual carried exactly in layers 1-6 only (protect policy, later layers trimmed to the arena schedule)
r35 host (F2J off) 7.911e-8 vs deployed host 8.593e-8 = **−7.9% raw**, u_local 0.702 (+17% util from 6144 extra CP terms pushed through the remaining depth). F2J-on 6.23e-8 is still the tape-confounded number (run predates the rank-feature fix). The deep-layer residual terms (L7+) were what hurt; early residuals carried exactly through the whole depth help. Queued: window map (1-3/1-4/1-8/1-10), net 100 frozen check, top-k residual rows (256/512 of 1024) for the bill, F2J-on with the fixed tape, post-hoc grading of the window planes.
Window L8-14 only (protect): host 1.537e-7 (+79%), u 0.622 — the deep residual carry alone is strongly harmful; the gain is entirely an early-layer (L1-6) effect carried through depth.

## 8. Window L1-6 with the FIXED F2J tape, and its post-hoc grading (r35)
| | host (F2J off) | full card (F2J on, deployed payload) | u_local |
| deployed | 8.593e-8 | 2.6758e-8 | 0.600 |
| win6 (residual exact L1-6, protect) | **7.911e-8 (−7.9%)** | 2.7795e-8 (+3.9%) | 0.702 |
| merge full depth (tape fixed) | 1.127e-7 (+31%) | 5.84e-8 (+118%) | 0.702 |
The host gain is real and the learned F2J (fit on the OLD host's residual pattern) mis-corrects the improved host: +3.9% on the full card. Realizing the −7.9% on the board therefore needs an F2J refit on the win6 host (frozen joint-refit protocol: 200 card runs × ~75 s ≈ 4.2 h CPU + minutes of fitting; `public_deploy_composition_prep_v1/run_joint_refit.py` lineage), and the bill (+17% util) must be cut (top-k residual rows, chain6) for the adjusted score to move: adj = raw × util → at u 0.702 a −7.9% raw is +7.7% adj WORSE; break-even needs the residual carrier below ~+8% util or a larger raw gain after refit.
Post-hoc grading of the win6 planes (tangent → consumption): only_v +1.0% at β 0.3 (dv_corr +0.41/+0.40 at the early injection layers, ≤0 deeper) — the linear v-channel response captures only a small part of the in-host −7.9%: the host's gain comes through its nonlinear couplings (k3/k4 targets, CP4/stars), not through the v-channel alone. The two routes (post-hoc supply vs in-host carry) are different objects; the in-host one is the larger lever here.

## 9. Window map (protect policy; host = F2J off vs deployed host; full card = F2J on with the DEPLOYED payload vs deployed 2.676e-8 r35 / 2.746e-8 r100)
| window | net | host MSE (Δ) | full card F2J-on (Δ) | u_local |
| 1-3 | r35 | 8.296e-8 (−3.4%) | 2.898e-8 (+8.3%) | 0.672 |
| 1-4 | r35 | 8.187e-8 (−4.7%) | 2.895e-8 (+8.2%) | 0.687 |
| 1-6 | r35 | 7.911e-8 (−7.9%) | 2.780e-8 (+3.9%) | 0.702 |
| 1-8 | r35 | 9.359e-8 (+8.9%) | 3.127e-8 (+16.9%) | 0.702 |
| 1-10 | r35 | 1.200e-7 (+40%) | 3.919e-8 (+46%) | 0.702 |
| 8-14 | r35 | 1.537e-7 (+79%) | — | 0.622 |
| 1-6 | r100 | 7.408e-8 (−5.0%) | 3.242e-8 (+18.1%) | 0.728 |
The host gain peaks at window 1-6 and transfers with the same sign to r100 (−5.0%); the deployed F2J payload mis-corrects every retrofitted host (its residual pattern changed), so with the learned operator kept unrefit the in-card route is net negative on the full card — second priority, as the coordinator ruled (T-24/T-25).

## 10. (B) DIRECT SIDECAR — emission delta on the card's OWN state, one-layer slice push, no carry (`build_sidecar_planes.py`, SIDECAR_DIAG_rank*.json)
E_L = engine `emission_K21`(host m,v,k3,k4,C,K21,K22,K31) − card `_full_relu_reset` K21 target on the SAME inputs (engine function exec'd verbatim from run_tensor_recursion_v4.py; card function from the retro copy). r35, off-diagonal rms:
| next layer | E_L (post) | pushed Δpre | card plane | truth−card gap | corr(Δpre, gap) |
| 2 | 0.0e+0 | 0.0 | 1.20e-2 | 5.7e-4 | −0.01 |
| 3 | 1e-5 | 1e-5 | 9.3e-3 | 5.5e-4 | +0.05 |
| 4 | 1e-5 | 1e-5 | 7.6e-3 | 5.3e-4 | +0.04 |
The two emissions agree to ~0.1% of the plane; the pushed delta is ~50× below the truth gap and uncorrelated with it. VERDICT: the emission is NOT where the engine's value is — the engine's +10.6/+15.3 lives in the CARRY (the exact tensor carry of the post plane vs the card's one-step star). Consistent with §4 (exact residual carry moves the card planes toward the engine's) and §7-9 (the in-host carry gain). Tangent → consumption confirmation of ≈0 appended below when it lands.

## 11. (B) confirmed through tangent → consumption, and the cost ladder (F2J-on vs deployed)
Direct sidecar (emission delta on the card's own state, no carry): r35 only_v −1.3/−0.9/−0.8/−0.6/−0.4/−0.3/−0.1 % at β 0.82/0.6/0.5/0.4/0.3/0.2/0.1 (dv_corr ≈ 0: +0.03, −0.07, −0.05, −0.04, −0.15); r100 0.0 at every β. E_L stays 1-3e-5 rms at all layers on r100 too (SIDECAR_DIAG_rank0100.json). The engine's +10.6/+15.3 is entirely in the carry.
Cost ladder for the in-host window-6 carrier (r35, host F2J off vs 8.593e-8; full card F2J on vs 2.676e-8): top-256 residual rows → host 1.453e-7 (+69%), F2J-on 7.01e-8 (+162%), u 0.633; top-512 → host 1.227e-7 (+43%), F2J-on 5.22e-8 (+95%), u 0.662; scale 0.5 → {'u_local': 0.7016, 'mse_nc_f2j_on': 2.6577817608699306e-08, 'mse_nc_f2j_off': 7.991706675140552e-08, 'raw_delta_pct_f2j_on_vs_dep': -0.67, 'raw_delta_pct_f2j_off_vs_dep': 198.67}. Partial residual sets are worse than none: the residual rows are a coherent plane object and cannot be thinned to cut the +17% bill.

## 12. Verdict for the coordinator (T-24/T-25 frame; every full-card number is F2J-on vs deployed)
| route | r35 full card vs 2.676e-8 | r100 full card vs 2.746e-8 | util | status |
| (A) window-6 planes as post-hoc supply (tangent → consumption) | only_v +1.0% (β 0.3) | — | +17% if the carrier is billed | small |
| (B) direct emission-delta sidecar, no carry | −0.1 … −1.3% | 0.0% | ~free | DEAD: emissions identical to 1e-5 |
| (C) in-host window-6 carry, deployed F2J kept | +3.9% (host −7.9%) | +18.1% (host −5.0%) | +17% | net negative without an F2J refit |
| in-host window-6 + top-k residual rows (256/512) | +162% / +95% | — | +5% / +10% | dead |
| in-host window-6 at HALF strength (scale 0.5), deployed F2J kept | **−0.67%** (host −7.0%) | — | +17% | the only full-card improvement; adj-negative at this bill (raw −0.7% × util +17%) |
What the whole set says: the engine's value is the exact carry of the post-K21 plane through depth; inside the card that carry helps the host (−7.9%/−5.0%) only in the early window, costs +17% util, and is undone by the deployed F2J payload; as a post-hoc plane supply it is worth ~+1% because the arena schedule truncates it at depth and the linear v-channel response captures little of the in-host gain. Nothing here reaches board-positive at the fire bar without (i) an F2J refit on the window-6 host (≈4.2 h CPU, frozen protocol) AND (ii) a cheaper carrier — and the cost ladder says the residual cannot be thinned.
