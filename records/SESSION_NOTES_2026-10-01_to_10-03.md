# Opus re-examination session — started 2026-10-01 17:00 UTC (lead: Claude Opus 5.5)

P2 = `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16` (all paths below are relative to P2 unless absolute).
H = `P2/codex_resume_20260926_v2`. This directory = `P2/opus_reexam_20261001_v1` (call it O).

## 1. Where we stand (measured 2026-10-01)

Score per network = final-layer MSE x max(0.1, billed flops / 2^41); the board is the mean over networks; lower is better.

| team | adjusted score | final-layer MSE ("raw") | all-layer MSE | share of budget used | note |
|---|---|---|---|---|---|
| J2W #333322 | 1.629e-9 | 1.483e-8 | 6.59e-9 | 0.110 | wall 95 s mean / 104 s max |
| marius_binner #333355 | 1.652e-9 | 1.140e-8 | 4.75e-9 | 0.145 | raw fell 1.556e-8 -> 1.14e-8 in four days of small steps at a constant bill |
| Luna #333152 | 1.844e-9 | 1.412e-8 | 5.75e-9 | 0.131 | 16 entries in total |
| suliman_tadros #332755 | 2.048e-9 | 1.578e-8 | 6.40e-9 | 0.130 | |
| **us, S35 #333111** | **2.067e-9** | **1.712e-8** | 6.78e-9 | **0.121** | wall 82 s mean / 95 s max, residual 0.34 s max |

- Bank source: `H/hill_climb_prefix128_fusion_v1/candidate.py` (sha f9f4790b...). Research parent (same predictions, bill -0.10%):
  `H/hill_climb_pair_rr_inverse_v1/candidate.py` (sha a3efb3a1..., "RRPI").
- Next bank: 2e-9 needs -3.3% adjusted. Podium (below Luna) needs about -11%. 1e-9 needs -52%.
- We have the worst raw of the top five and the second-lowest bill. Every leader's per-network error correlates 0.72-0.85 with ours:
  same method family.
- The board's 100 evaluation networks are NOT in the public dataset (checked by name), so board gains are honest generalisation.
- Competition closes 17 Oct 2026 23:59 UTC; 10 submissions per day.
- Full public history of every team: `O/board_evals/scan_332296_333380.jsonl` (tool: `O/board_evals/scan_submissions.py`).

## 2. What the estimator is (one paragraph)

Layer by layer it carries the mean m, the full covariance C (1024x1024), and the third-order cumulant tensor as (a) a bank of
symmetric rank-one columns p_r with weights lam_r (two columns per "kink" unit born at each layer, pushed through W every layer;
old cohorts are compressed into rank 384/320/384 frames at layers 7/10/13), (b) a vector b for the collective gain mode
Sym(C x b), (c) a plane-only remainder ("star"). Fourth order is one scalar per layer ("radial"): k4 = radial v^2,
K22 = radial (v v^T + 2 C o C)/3, K31 = a per-row form. A pair map turns (mu, C_pre, K21, k4, K22) into the post mean and
covariance. Learned parts: a small per-unit message-passing network (45k parameters) that corrects the mean at each layer, learned
birth coefficients, a small learned covariance correction at layers 8-14, and about ten physics dials.
Bill (S32 census, `H/hill_climb_s32_bill_census_v1/summary.json`): column push 43%, plane read 34%, covariance + star 11%,
final-layer k3 3%, pair map 2%, learned parts 3%. Everything large runs through depth-6/7 Winograd products.

## 3. Measurements made this session (network 35 unless stated)

Tools in O: `e1_gain_conditioning_mc.py` + `e1_analyze.py` (Monte Carlo law of total cumulance), `e2_dump_card_cumulants.py`
(torch S6b card dump), `e3_card_vs_truth_split.py`, `e4_kurt_transport_test.py`.

1. **Per-layer error profile** of the current parent (x1e-9, layers 0..15, net 35): 0.69 0.60 0.81 1.36 2.05 2.97 4.17 4.55 5.97
   6.87 8.74 10.94 11.98 13.78 14.45 14.60. Roughly +1e-9 per layer; layer 0 is pure truth noise.
2. **Where the fresh error of each layer comes from** (torch S6b card vs 32M-sample truth, mean-formula sensitivities): at layers
   10-15 the card's per-unit input errors inject about 2e-9 (variance, relative error 0.1-0.2%), 1.2-1.5e-9 (skew, error 5-9% of
   the skew spread) and 1.1-1.5e-9 (kurtosis, the card has no per-unit variation at all) of mean-squared error per layer. A carried
   mean error shrinks by about 0.75 per layer, so the final error is roughly 4x the per-layer injection of the deep layers.
3. **Collective gain mode.** Conditioning each layer's pre-activations on the norm of the previous layer removes ALL of the mean
   excess kurtosis (0.06 at layer 15) and 87% of the per-unit skew variance (skew spread 0.102 -> 0.037). What is left is
   network-specific: per-unit kurtosis spread 0.015 around zero, skew spread 0.037-0.044.
4. **The card's skew error is not a mis-scaled gain mode.** Its error is uncorrelated with both the gain part and the bulk part
   (R^2 <= 0.10). The card's b carrier holds only 20-30% of the gain part; the bank + star hold the rest (correlation 0.995).
   The card's variance error is not a gain-variance amplitude error either (R^2 on (mu/sigma)^2 <= 0.1).
5. **No per-unit carrier can transport the network-specific kurtosis or skew.** Even the TRUE per-unit kurtosis at layer l,
   pushed through variance-share, coincident-index or elasticity-dressed operators, predicts layer l+1 with R^2 <= 0.16 (skew:
   R^2 ~ 0). The information lives in pair-level (distinct-index) fourth-order structure.

## 4. Conclusions of the re-examination

- The remaining error is network-specific third/fourth-order content that the card either approximates isotropically (fourth
  order) or carries with 5-9% error (third order). Completing fourth order exactly needs pair-level carriers at roughly +50-80%
  of the bill; the earlier team's "economically closed" verdict survives every shortcut tried today (single collective factor,
  per-unit volatility vectors, youngest-cohort-only terms, Monte Carlo hybrids of any kind).
- Monte Carlo cannot help at this accuracy: 6,500 full-depth samples cost the whole 10% floor and give 1e-5 per-unit error
  against the card's 1.3e-4... (variance 0.07/N); no control variate exists because the closure is not a per-sample surrogate.
- The one hard fact about the leaders: marius_binner improved 27% in raw at a constant bill through many small steps. That is
  what continued training of learned parameters looks like. Our refits were each a few hundred optimiser steps.
  The current torch trainer costs 26 s per network gradient on one thread, so thousands of steps per day are affordable.

## 5. Lanes (2026-10-01)

- **trainer**: parallel long-horizon refit of all learned groups on the current card (640 training networks), export, native validation.
- **engineer**: bill reduction on the research parent, target -3.5% adjusted at neutral raw, for the 2e-9 bank.
- **analyst**: extend the diagnostics above to the current card and more networks, track the leaders daily, build the explainer.
- Lead: coordination, native validation of every candidate, board submissions (only the lead submits).

## 6. Machine rules for this session

16 cores, 48 GB, shared. Before any launch above 3 GB run `bash O/gate.sh <own peak GB> <label> &&`. Heavy-process budget:
trainer at most 3 workers (about 6.4 GB each), engineer 1 native job (about 6 GB), analyst 1 job. One MPS (GPU) job on the box at
a time. `nice -n 10`, explicit thread counts. Never edit or delete existing files outside your own lane directory; version instead.
No git commits. Only the lead submits to the board.

## 7. Timeout risk from the public record (added 18:50 UTC)

From 957 graded phase-2 runs of all teams (`board_evals/scan_332296_333380.jsonl`): share of runs that lost at least one network to
the 120 s limit, by the run's MEAN predict wall: below 60 s 0/275; 60-75 s 2%; 75-85 s 2%; 85-90 s 9%; 90-95 s 13%; 95-100 s 12%;
100-105 s 21%; above 105 s 51%. One lost network ruins the score (ours: #333099, mean wall 88 s, four networks on one slow worker,
score 0.0074). The bank S35 runs at 82.5 s mean / 94.9 s max. The private re-evaluation scores ONE designated submission, so the
designated file should run at or below about 75 s mean. Consequence: CPU reductions carry insurance value on top of their bill
value, and anything that adds CPU is suspect even if it saves bill. Rivals at risk: J2W (95 s mean) and Luna (90 s mean) about 13%
per run; marius 82-89 s; suliman 81 s.

## 8. Additions 19:00 UTC

- Fourth lane **release** (O/release/): qualification + packaging runbook, platform wall/residual projection, CPU anatomy, a faster
  "safe" variant for the designated submission. Never uploads.
- Machine budget with four lanes: trainer 2 workers (~13 GB), engineer 1 native job, release 1 native/boundary job, analyst 1 job.
- E5 (`e5_scale_mixture_planes.py`, net 35, layers 5/9/12, two independent 8M truth draws): the collective-gain (scale-mixture)
  identities hold in structure (the fitted gain carrier is 1.0-1.4x radial/6 times the mean; 69-78% of the kink-row K21 plane energy is
  the gain part) but explain the true K31 plane only a little better than the card's form: share of the reachable variance explained
  0.38-0.55 (card form) -> 0.46-0.60 (bulk form + gain-skew terms). About half of K31 at kink rows is network-specific bulk
  fourth-order content. K22 is 95-99% isotropic already. So there is no free K31 formula worth more than about a point.
- E6 (`e6_gain_elasticity_mc.py`, net 35, 2M samples, 192 units at layers 6/10/14): how conditional moments respond to the squared norm
  of an earlier layer. Means: elasticity 0.50-0.52 (rigid scaling predicts 0.5). Per-unit variances: 0.81-0.83 against the previous
  layer's norm, 0.87-0.96 against the norm six layers back, 1.00 against the input norm. Off-diagonal covariances (energy-weighted):
  0.91-0.94, 0.94-0.98, 1.00-1.02. So the collective gain mode IS close to a rigid scale mixture for second moments, including pair
  covariances; the card's Sym(C x b) form has the right structure. Together with E3 (the card's skew error is uncorrelated with the gain
  part) this means the gain mode is not where the error is.
- Extra check: the card's per-unit variance error is not leakage of its carried mean error (correlation of dv with -2 mu dmu between
  -0.05 and +0.18 at layers 4-15).
- Torch training card vs native (analyst): the torch card omits six native cost-saving approximations and is 0.7-3.4% more accurate than
  the native estimator on networks 35/37/39; error vectors correlate 0.99. Trained parameters are therefore tuned on a slightly
  different card than the one that is scored.

## 9. E7 (19:05 UTC): the learned mean correction is information-limited, not under-trained

`e7/e7_dump.py` + `e7/e7_fit.py`, current torch card, 25 training + 24 validation networks, layers 4-15. Target = the host error
freshly made at each layer (host minus truth minus the previous layer's error pushed through the gate). On the 24 held-out networks
the deployed correction removes 10-15% of that error's energy at every layer; ridge regression and gradient boosting fitted on the
same inputs (the 30 features and the 64-number state) remove 10-15% as well; boosting fitted on what the deployed correction leaves
finds 0-3%. Summed over layers 4-15 the fresh host error is 31.8e-9, 28.0e-9 after the deployed correction, 27.8e-9 after boosting.
Consequence: more optimiser steps cannot buy more than a fraction of a percent; the trainer's pilot stays as a confirmation only.
The e7 pair of scripts is now a fast screen (minutes) for any candidate per-unit feature: dump it, refit, read the held-out share.

## 10. Covariance error anatomy and closed-loop tests (19:15-19:35 UTC)

Scripts `e8_recenter_and_cov_oracle.py`, `e8b_recenter_scan.py`; analyst tables `analyst/a4_*`, `analyst/a6_*` (`analyst/RESULT.md`).

- Analyst A6 (four networks, current torch card, learned parts on): making the per-unit VARIANCE fed to the mean formula exact at
  every layer lowers the final-layer error by 34% (25-43%); kurtosis 7%; skew 0%; variance + kurtosis 50%; all three 67%. With the
  learned correction off, all three exact gives 0.8e-9 (from 17.2e-9): the whole error is in the three per-unit inputs.
- The card's carried covariance error E (network 35, layers 6/10/13, 32M-sample truth): relative off-diagonal error 0.49/0.72/0.92%.
  With mh = m/|m|: E = alpha mh mh' + (mh d' + d mh') + Eperp. Shares of |E|^2: alpha term 3-10%, cross term 19-48%, Eperp 48-71%.
  The cross term carries 20-42% of the next layer's per-unit variance error energy.
- Closed loop (network 35, truth removed at layers 2-14, learned on, no refit): removing the alpha term +0.1%; the cross term 0 to
  +1.3%; both +0.2 to +1.6%; Eperp alone -33% (half strength) / -34% (full); all of E -30% / -29%. So the mean-aligned parts are not
  a lever; the whole covariance lever is the part orthogonal to the mean. Same law as on the September card.
- Recentring strength (the card shifts its covariance by -kappa (host d' + d host' + d d') after the learned mean correction d,
  kappa = 0.92): the covariance data say about 85% of that shift is wrong (regression slope of the cross vector on d: -0.75 to
  -0.83), but the final error does not care: on 8 validation networks kappa 0.7 / 0.5 / 0.3 give +0.12% / +0.66% / +1.54% (network
  35 alone had shown -1%). The dial stays at 0.92. No free gain.
- A per-pair learned covariance corrector was already tested by the September team (held-out R^2 <= 0.02-0.09 from every local
  feature family; `fable_covhunter_v1/RESULT.md` section 19). Not repeated.

## 11. New structural idea under test (from 19:40 UTC): the gain-normalised frame ("frame B")

A ReLU network without biases is homogeneous: scaling the input scales every activation. The collective gain (the size of the
activation vector along its mean direction, relative sd 3.5% per layer, accumulating to about 9% by layer 14) can therefore be
divided out exactly at every layer instead of being tracked through third- and fourth-order cumulants:
  G_l = m_l . x_l / |m_l|^2,   u_l = x_l / G_l,   frame-B expectation = E[G_l f(u_l)],   E[x_{l+1}] = E_B[relu(W' u_l)]  (exact).
The carried mean is unchanged (the true mean). To leading order the frame-B covariance and third cumulant are the laboratory ones
projected orthogonally to the mean (C -> P C P with P = I - mh mh'), plus a small term b' = -2 P C m / |m|^2 of the Sym(C x b) form
that the card already knows how to carry, and no fourth-order (radial) machinery at all.
Evidence so far (`e9_frameB_formula.py`, network 35, exact per-unit inputs from 16M samples): in frame B the mean formula without
any fourth cumulant is as accurate as the laboratory formula with the exact fourth cumulant (layer 15: 6.3e-9 vs 4.8e-9 noise floor;
laboratory without fourth cumulant 24e-9). The third cumulant is still needed in both frames.
Why it might matter: 85-89% of the laboratory third-order plane at layers 11-14 is gain content, and the card's distinct-content
error lines up with gain-sector terms (September findings). In frame B that content does not exist, so closure errors proportional to
it would vanish. Why it might not: the harmful part of the card's error is uncorrelated with the gain part (E3).
Test plan: (a) Monte Carlo truth of frame-B covariances (`e10_frameB_mc.py`), (b) numerical certificate of the re-normalisation
formula, (c) a hook-level frame-B version of the torch card, physics only (learned correction off), against the laboratory card
under the same setting (17.2e-9 on networks 35/100/37/39).

### 11b. Frame B: first prototype is NEGATIVE (20:15 UTC) — parked

Files: `e10_frameB_mc.py` (Monte Carlo truth, network 35, 16M samples, `e10_rank0035_N16M.npz`), `e10_analyze.py`,
`e11_frameB_card.py` (hook-level frame-B version of the torch card), `e11_rank0035.json`.
- Monte Carlo facts (network 35). The per-layer ("fresh") variance of the gain coordinate t falls from 1.9e-3 after layer 1 to
  2e-4 after layers 14-15 (sum 8.4e-3 = the accumulated gain variance). Dividing by the projection on the mean does NOT leave a
  Gaussian state: the mean excess kurtosis of the normalised pre-activations is +0.0006 at layer 1, +0.007 at layer 2 and
  +0.012 to +0.019 at layers 6-15 (laboratory frame 0.009 -> 0.06). Dividing by the norm instead gives -0.008 at layer 1 and
  about 0 at layer 15: the "kurtosis vanishes" result of E1 was partly a cancellation with the sphere constraint (-6/n).
- The re-normalisation needs the third cumulant contracted with the mean direction: the covariances respond to the gain
  coordinate with a coefficient eps of 1.0-1.7 x var(t) (2.0 at layer 0), and the non-rigid part of that response is 0.15-0.4% of
  the off-diagonal covariance, so it cannot be dropped. Computed exactly from the bank it costs one extra full product per layer.
- The fresh gain is not an independent multiplier (it is a sum of the units themselves), so the simple transform of the
  fourth-order coefficient (radial' = radial + 12 var(t) - 12 eps) is wrong: it gives -0.020 where the truth is +0.0006.
- Closed loop, network 35, learned mean correction off, final-layer error: laboratory card 16.3e-9 (pure physics, no fitted birth
  corrections, no learned covariance edge: 22.1e-9). Frame-B card with the fourth-order coefficient taken from Monte Carlo (15
  oracle scalars): 39.4e-9 (pure physics 39.7e-9). Carried covariance error against its own truth, noise-corrected: frame B
  0.54 / 0.83 / 1.11% at layers 6 / 10 / 12 against 0.49 / 0.69 / 0.80% for the laboratory card (part orthogonal to the mean).
- Reading: removing the gain does not make the third-order planes or the covariance more accurate; the harmful error is not a
  by-product of the gain bookkeeping (consistent with E3). A correct frame-B card would need its own fourth-order closure, a
  projected star, and re-fitted birth corrections, for no demonstrated gain. Parked. What survives: the gain sector is confirmed
  harmless (section 10), and births without the fitted corrections cost +31% (22.1 vs 16.8e-9), i.e. the fitted birth corrections
  are worth about a quarter of the physics-only error.

## 12. E12 (20:40 UTC): the per-unit variance error is partly predictable, and that part is already used

`e12_dv_screen.py`, `e12b_dv_closed_loop.py` (current torch card; networks 35, 37, 39, which have both the learned-net input dumps
and 48-64M-sample truths of the per-unit pre-activation variance).
- Leave-one-network-out, target = relative error of the card's per-unit variance (rms 0.03% at layer 4 to 0.18% at layer 15, truth
  noise 0.018%): ridge and gradient boosting on the learned net's 30 inputs + 64-number state (+ a few extras) explain 9-20% of it
  at layers 5-15 on the held-out network.
- Closed loop on network 35 (predictor trained on 37 and 39; corrected variance fed to the mean formula at layers 4-15):
  learned mean correction ON: +4.9% (boosting), +1.2% (half strength), +5.3% (ridge) -> WORSE; truth variance: -31.6%.
  learned mean correction OFF: -1.6% / -1.7% / -5.0%; truth variance: -26.5%.
- Reading: the predictable part of the variance error is the same information the learned mean correction already uses (third
  confirmation after E7 and the September audits). No per-unit head, of any kind, adds anything.

## 13. Status of the lanes (20:45 UTC) and artifacts

- Explainer 47 (analyst) published for Matt, private: https://claude.ai/artifact/K5heSDnE8CDkBvJ91BaPMJ ("Remaining Error Anatomy").
- Release: pipeline + calibration done (`release/RESULT.md`); a slow worker also breaks the 0.4 s residual limit (three of #333099's
  four lost networks); surviving x1.30 needs <= 38,300 requests per network (S35: 42,200). Task now: the safe variant (depth-7 off
  + request cut; targets mean wall <= 75 s, residual <= 0.275 s).
- Engineer: bill attempts (response floor, select64, alternative-basis Strassen kernels "hzs", rounding); status requested.
- Trainer: pilot A at update ~16/106; first validation number pending; capped at 2 workers.
- Analyst: A10 leader forensics per network, periodic board scans, standby for native validation.
- Fifth teammate: "physicist" (O/physicist/), from 20:55 UTC: one-step injection budget on the current card, then missing birth / fourth-order terms; candidates need shadow value <= -1.5% at < 0.5% bill. (An "explainer" teammate was started at 20:47 and stopped at 20:52 on Matt's instruction: 100% of effort goes to score improvements.)

## 14. Evening findings (21:00-22:00 UTC)

- **Trainer verdict** (`trainer/RESULT.md`): the learned parameters are not optimisation-limited. Pilot A on 20 validation networks:
  -0.36% after 18 updates, -0.04% after 23 (moving average -0.37% / -0.27%, se 0.2-0.3%); an exact least-squares refit of the
  final-layer correction gives -0.24% on unseen training networks and +0.02% on validation. Expected value of more training of
  the existing parameters: 0.2-0.4%. Pilot stopped; the trainer prepares run B (per-layer physics dials, learned nets frozen).
- **Engineer** (`engineer/a04_hzs_all`): half-zero-strip padding kernels: bill -1.22%, raw +0.18% / +0.23% on the 4 screen and 8
  check networks = -1.0% adjusted. Plain products instead of fast multiplication change the pooled raw by +0.08% (per network
  +-0.4%), so rounding in the kernels costs nothing systematic. Response floor 0.0075: raw -0.8%, bill +0.7% (neutral).
- **Release** (`release/RESULT.md`): level fusion restricted to left-operand trees: bill 0, bytes identical, projected platform
  kernel time -13%. Safe variant 'fastC' (that + thin star products as plain matmuls): bill +0.37%, 73-74 s mean wall, residual
  0.283-0.287 s, stages A, C, D passed. The platform's setup spends its whole 4.0 s guard binding and fuses only the 128 eager keys.
- **Physicist T1** (`physicist/t1_onestep.py`): one-step injection of the carried covariance on the current card (networks 35, 100):
  K21 plane at kink rows 63-74%, K31 18-26%, K21 other rows / K22 / per-unit k3, k4 about 5% each or less.
- **Physicist T2** (`physicist/t2_budget.py`, Monte Carlo truth of each mechanism of the post third cumulant, network 35): the card's
  kink-block plane error is 4.5% (layer 6) to 7.4% (layer 14) of the true plane; 50-79% of its square is CARRIED distinct-index
  content (accumulated), and the fresh part is dominated by the FOURTH-CUMULANT FEED into births, (d_c/2) g_i g_j Q_ijcc: the true
  feed is 3.2-3.3% of the plane norm, the card's isotropic version explains 21-27% of it (fitted coefficients: the C_ij v_c part at
  0.75 of radial/3, the C_ic C_jc part at about 0, where the card uses radial/3 and 2 radial/3). Third-cumulant insertions (the
  birth legs) leave 1.3%; Gaussian births 0.6%; the star 0.6%.
- **E14** (`e14_tborn_fourth.py`, network 35, layers 5 / 9 / 12): the fourth cumulant born from the third cumulant at one layer
  (hub with one third-cumulant insertion and one covariance edge, computed from the card's bank) is physically right and small:
  added to the isotropic form it raises the explained variance of the true K31 at kink rows from 0.41 / 0.38 / 0.32 to
  0.51 / 0.48 / 0.44 (fitted coefficient 0.87-0.89, i.e. about 1), and it explains 12-16% of the network-specific per-unit kurtosis
  (correlation 0.34-0.40 with the residual, coefficient 0.70-0.86). The per-unit part needs one extra reduced read per layer
  (about +7.5% bill) for about 14% of a 7% lever: not worth it. The rest of the fourth-order content is older than one layer, and
  carrying it costs at least as much as the third-order bank.
- **Consequence**: the dominant fresh error (fourth-cumulant feed into births) and the kurtosis error both need a fourth-order
  carrier; within this card the remaining capacity is co-adapted retuning (run B) and the bill.
- Running: `e13_layer_knob_fd.py` (finite-difference Jacobian of the final error for 91 per-layer scalars on 8 networks) for a
  Gauss-Newton estimate of what per-layer dials can give (`e13_fit.py`).

## 15. 22:15 UTC: validation bias, per-layer dials (E13), plan

- **The 40 validation networks are a biased judge for learned parameters** (trainer, `trainer/RESULT.md`): the incumbent parameters
  were tuned on the old 160 training networks and selected on those 40. On 60 newer-rank networks nobody trained on, pilot A's
  moving average gives -0.10% (se 0.11) after 18 updates and -0.43% (se 0.15) after 23; on the 40 it shows -0.04 to -0.36%.
  The same bias explains the engineer's observation that every rounding-level kernel variant moves the pooled error of the screen /
  check networks up by 0.02-0.23% (19 of 24 per-network changes positive): the incumbent was selected on them. On fresh networks
  (the board) a rounding-level change is neutral, so the padding kernels are worth their full bill change (-1.22%).
  RULE from now: learned-parameter candidates are judged on the trainer's 100 held-out newer-rank networks
  (`trainer/cert/runB_val.txt`), kernel-only candidates on bill and byte/rounding parity.
- **Run L** (trainer): the long run continues from pilot A's checkpoint after 25 updates with 541 training networks and 100
  held out for good; first bank candidate at a held-out gain of -0.6% or better.
- **E13 per-layer dials, negative** (`e13_layer_knob_fd.py`, `e13_fit.py`; finite-difference Jacobians of the final error for 91
  per-layer scalars in 7 families on networks 35, 36, 37, 39, 100; Gauss-Newton fit): in sample -3.8 to -5.5%, leave-one-network-out
  +3.7 to +10.9% (0 of 5 networks better); single families and even one scalar per family do not transfer either (+0.3 to +2%).
  The knob effects on the final error are network-specific; a handful of networks cannot find the systematic part. Per-layer dials
  stay a low-priority item for the trainer (exact gradients on 541 networks), expected gain small.

## 16. 23:10 UTC: F4, the first physics gain of the day (physicist), and the bank plan

- **F4** (`physicist/f4.py`, `physicist/t3_screen.py` option cp4): every bank column stores one scalar from its birth,
  A = -alpha / (sigma (1 + radial/3)) of its hub unit, i.e. the third derivative coefficient of the ReLU at the hub over the birth
  weight; this makes the columns carry the Gaussian-born fourth-order "star" (hub with three smooth legs) as Q = sum_r theta lam_r A_r p_r^4
  at no cost, and its feed into the next births, (d_i/2) Q_ii.. , is an IN-PLACE SHIFT of the young carried columns after the gate:
  g o pp -> g o pp + theta A (d/2) o pp o pp (columns at most two layers old), with a cheap anchor that removes the plane of the
  added tensor from the star (two matrix-vector passes + a diagonal). No new columns, no dense product: estimated +0.2-0.25% bill.
- Torch, learned parts on, NO refit, final-layer error: four networks -2.33% at theta 0.4 (learned mean correction off -4.07%; shadow
  estimate of a refit -2.85%); 40 validation networks: theta 0.2 -1.67% [95% -1.99, -1.34], 39/40 better; theta 0.4 -2.46%
  [-3.03, -1.85], 35/40 better, all-layer -2.22%. Pending: strengths 0.6-1.0, 24 held-out newer ranks, age beyond two layers,
  the free per-unit kurtosis from the same weights, the port spec (`physicist/F4_SPEC.md`).
- This is the September "fourth-order weights on the bank's columns" idea used where it is free: September priced it for the
  K31 / K22 READS (a full extra read each) and closed it; the birth FEED needs no read, only a column shift.
- **Engineer**: full stack `engineer/stack/candidate_T_C_B_F_D_Hfull.py` = float32 fixes + half-zero strips: bill -1.957% on the 12
  old networks, -1.863% on 10 fresh networks (ranks 660-669) with pooled raw -0.107% there (8/10 lower): the +0.2% on the old
  networks was RRPI's selected rounding draw (noise decomposition: RRPI's rounding noise is aligned with its error by -0.15% on the
  old networks, the variants' by about 0). Regenerated prebind list from a fresh census: lazy-bind bill 0.46% -> 0.32% (pending).
- **Trainer**: per-layer dials certified (217 scalars, gradients recorded for free in run L; on four networks nothing beyond noise).
  Run L after 25 updates: -0.237% (se 0.109%) on the 100 held-out networks.
- **Analyst**: fresh training networks in production (seed 7,100,000 + k, 64M samples, 155-180 per day, final-layer truth noise
  1.05-1.33e-9; certificate against the organizers' truth passed; an MPS random-number defect (the value -14.372 once per 8.4M draws)
  found and masked).
- **Bank plan**: S36 = RRPI + restricted level fusion + engineer's full stack (`engineer/stack/candidate_T_C_B_F_D_Hfull_Rlvfuse_sized32.py`),
  bill about -1.96% against S35 on fresh networks, raw neutral: expected 2.027e-9; the release lane is qualifying it (chain 9b:
  boundary run under the platform's setup clock, slow-worker stress, then the 20-network worker and the fresh-network native pair).
  S37 = S36 + F4 (native port by the engineer): expected about 1.98e-9. Then F4 extensions and run L parameters.

## 17. E15 (2026-10-02 00:25 UTC): anatomy of the fourth-cumulant feed from first principles (open loop, network 35, layers 9 and 13)

Scripts `e15_local_feed.py`, `e15b_collective_feed.py`, `e15c_exchange_feed.py`, `e15d_gate_feed.py`, `e15_capture_cols.py`; logs `e15*_rank0035_L{9,13}.log`.
Target = the physicist's true feed (Monte Carlo, two independent 8M blocks), pushed distinct content on the kink block of the next layer; score =
share of its energy explained, cross-block; an isotropic term with a fitted coefficient is always present. Layer 9 / layer 13:

| model (coefficients at their physical value 1 unless "fitted") | L9 | L13 |
|---|---|---|
| isotropic only | 0.214 | 0.212 |
| + rule A of F4 (one fitted strength, 0.73) | 0.312 | - |
| + LOCAL: the four exact hub coefficients S1..S4 on every pair, all ages | 0.335 | 0.341 (fitted 0.90) |
| + LOCAL restricted to what two in-place shifts can carry (RIDE) | 0.333 | - |
| + LOCAL + collective drive with exact weights (COLLX) | 0.355 | 0.346 |
| + LOCAL + COLLX + TB1 + TB2 (s-channel exchange) | 0.432 | 0.415 |

- LOCAL: S_m = fourth cumulant of the post-activations with m legs on the hub = (4-m)-th derivative of the m-th post cumulant with respect to
  the pre-activation mean: S1 = E f''' = -alpha pdf/v, S2 = 2(g - g^2 - m d), S3 = 3(1-g)(E x^2 - 2m^2), S4 = kappa4(x); S0 = 0.
  Fitted coefficients: per age 0.92 / 0.94 / 0.83 / 0.76 (L9), per coefficient at age 1: 0.99 / 0.89 / 0.90 / 0.70. S2 matters as much as S1.
  Rule A (S1 only, the pair's two weights) was standing in for it: its coefficient falls to 0.13 once LOCAL is in the fit.
- TB1 (every column shifts by d o theta_r, theta_r(k) = sum_h P_hr d_h W_hk y_h(k): the covariance of the gates of consecutive layers,
  E[J] != G W G): fitted coefficient 1.004 (L9), 1.03 (L13); +3.5..3.9 points. Costs a second push of all columns. Its collective (top
  eigenmode) part is nothing (+0.1 point; the top mode holds only 17-33% of the off-diagonal energy of C), so no gate-shaped shortcut.
- TB2 (both columns of a newborn pair shift by (d/2) o Pi[., hub], Pi = the bank's plane between next-layer squares and previous-layer
  units): coefficient 0.86; +3 points; costs a plane-sized product.
- COLL (every column shifts along the same vector d/2 by c_r = (2/n) V21 . column one layer earlier): +0.8..1.5 points, free.
- 57% of the feed is still unexplained: the 8 other placements of the same exchange (no column form), older exchange content, higher order.
CONCLUSION: the tree-level theory is right term by term (coefficients 0.9-1.0 with nothing fitted), but the feed is many comparable
diagrams; only the hub-local part (12 points) is free, and F4's rule A already holds 10 of those 12. The cheap upgrade is F4b = the
RIDE shifts with the exact S1..S4 (sent to the physicist for a closed-loop screen). A perfect feed would be worth roughly -20..-35% of
the final error (scaling F4's -2.5..-4.5% for 10 points), so the lever is real but its remaining part is expensive: each further
3-4 points costs a push-sized product (+10..45% bill).

## 18. E16 (2026-10-02 00:57 UTC): K4L, a non-isotropic per-unit fourth cumulant from the exact hub coefficients. POSITIVE

Spec `K4L_SPEC.md`; scripts `e16_local_k4.py` (open loop), `e16b_k4_closed_loop.py` (closed loop; arms c, age, final, pair, f4, on, bsrc, k22),
`e16c_fresh24.sh`; logs `e16_rank0035.log`, `e16b_rank*_s1.log`, `e16b_rank*_s2.log`, `e16c_fresh24/`.
- The card's per-unit k4 is exactly radial v^2 (correlation with v^2 = 1.0); against 32M truth (network 35) its relative rms error is
  0.138 / 0.208 / 0.221 / 0.261 at layers 5 / 9 / 11 / 13 (truth noise 0.02-0.03); its level is 4-10% low (band dials 0.96-0.98).
- Hub terms summed elementwise over the pairs: norm 0.21-0.38 of the truth; added at coefficient 1 without removing their isotropic part
  they double count (worse); isotropic + hub terms, weighted by the mean formula's kurtosis weight: 0.28 / 0.26 / 0.25 / 0.34 of the
  residual energy explained, hub coefficient 0.88 / 0.89 / 0.88 / 1.02, isotropic coefficient -0.009..-0.013.
- Closed loop, learned on, no refit, final-layer MSE, networks 35 / 37 / 39 / 100:
  K4L age <= 2, c = 1: -2.69 / -2.94 / -1.96 / -3.65, pooled -2.83%; all ages c = 1: pooled -2.74% (39: +0.05%); all ages c = 0.5: -3.01%;
  F4 (theta 0.4): pooled -2.33%; F4 + K4L age <= 2: -7.43 / -3.64 / -8.24 / -3.16, pooled -5.50%; F4 + K4L all ages c = 0.5: -5.41%.
- Mechanism: the same exact local vertices that F4's rule A approximates for the feed, used where they are free: the diagonal.

## 19. 2026-10-02 01:00-02:30 UTC: fresh-rank validation of F4 and K4L, constants, S36 fired, S37 in qualification

- **Screen four are biased judges for physics levers too.** K4L alone: screen four -2.83% (4 of 4); validation 42/48/59/63: +1.2 / -2.3 / +2.5 / -4.8;
  24 held-out newer ranks (`e16c_fresh24/`): on top of F4 pooled -1.41% [95% -2.22,-0.59], 18 of 24 (age <= 4: -0.87% [-1.90,+0.25]).
  F4 (theta 0.4): -3.54% [-4.10,-2.94], 24 of 24 (the physicist's number exactly); F4 + K4L: -4.90% [-5.66,-4.13], 24 of 24.
- **Why K4L's final-layer effect has either sign per network.** It acts only through the first pair call (mean formula; the covariance
  recomputed by the knob hook does not change: `e16b_rank0424_s6.log`). Layers 2-6 improve on every network (-2..-9%; with the learned
  correction off -6..-15% at layers 1-5). At depth the kurtosis-related part of the mean error partly cancels the variance-related part
  (A6: exact variance -34%, exact kurtosis -7%, both -50% -> cross term about -9% of the base error), so removing a quarter of it is
  worth about -1.4% on average with a per-network sd of 2.3%.
- **F4's strength on fresh ranks** (`e16e_fresh24_theta/`): 0.6 against 0.4 -0.35% [-0.70,-0.00] (15 of 24 better); 0.8 against 0.4 +0.16%.
  The 0.4-0.48 of the screen four / VALID40 was depressed by the learned correction's compensation. Constant for S37: 0.6.
- **Check set nobody tunes on** = the analyst's generated networks (`e16g_generated/`, seeds 7100000+; base error mean 19.9e-9 incl. about
  1.2e-9 truth noise, sd 3.7e-9 against 1.8e-9 on the public ranks): F4 0.4 -3.2% (8 of 8), F4 0.6 + K4L -5.9% (8 of 8).
- **E18** (`e18_free_weight_anatomy.py`): the physicist's free per-column fourth-order weights (0.49 / 0.60 of the feed at layers 9 / 13)
  are a per-PAIR scalar (correlation between the + and - column of a pair 0.92-0.95) = 0.86 x rule A + a constant, R2 0.44 / 0.31
  (age 1: 0.70 / 0.64); no other cheap per-column quantity adds anything. The rest of the free fit is not predictable from hub data.
- **E17** (`e17_k31_local.py`): the K31 plane off the diagonal: isotropic R2 0.41 / 0.39 / 0.34 (layers 5 / 9 / 12); the per-unit row
  factor of K4L adds 0.003; exact hub planes add 0.07 at layer 5 (coefficient 0.88) but 0.013-0.015 at layers 9 and 12 (coefficient 0.4).
  The fitted isotropic coefficient is 0.76-0.81 x radial = radial minus the hub-local share (the fitted k31 knob -0.212, the trainer's
  b-source 0.85 and the physicist's feed coefficient 0.73-0.80 are the same fact).
- **Missing second-order term of the mean formula** (m = Gaussian + k3 E f'''/6 + k4 E f''''/24; no k3^2 E f^(6)/72): rms 1.6e-6 (layer 1)
  to 1.1e-5 (layer 13) against a per-layer error rms of 2.5e-5 to 1.2e-4: about 1% of the error energy; not pursued.
- **Per-layer error** grows linearly: 0.63e-9 (layers 0-1 = truth noise), then about +1.1e-9 per layer to 16-17e-9 at layer 15.
- **S36 fired 02:26:57Z, #333429** (stack + regenerated prebind list + un-fuse + thin-star plain matmuls; S35 physics): bill 0.98182x,
  expected about 2.02e-9. Fired on stage A + stage C + two networks of stage D (the box was 6x slow: 400-450 s per network).
- **S37** = S36 file + F4 0.6 + K4L (age 2, c 1): engineer's file sha 2a25a074, native parity for both terms on the screen four,
  bill +0.340%, +1,775 charged calls; stage A PASS; C and D running in a quiet window (`release/lead_s37_chain.sh`). Expected about 1.92e-9.
- **pgrep trap again**: my own waiter shell contained the names of the two release scripts and blocked my launcher's `pgrep -f` wait for 9 minutes.

## 20. 2026-10-02 02:30-04:00 UTC: S36 and S37 graded; three more negatives

- **S36 #333429 = 2.031e-9** (S35 2.0672e-9; -1.74%): error 1.712e-8 (unchanged), budget share 0.118611, wall 75.3 / 80.0 s, residual 0.283 / 0.314 s.
- **S37 #333435 = 1.951e-9** (-3.94% against S36, -5.62% against S35): error 1.639e-8 (-4.26%), all-layer 6.458e-9 (-4.7%), budget share
  0.119015, wall 78.0 / 87.6 s, residual 0.303 / 0.329 s, 100 of 100 networks. 4th on the board. Offline had said -5.2% (24 held-out ranks,
  torch), -6.0% (10 ranks native), -4.9% (13 generated networks, torch, diluted by truth noise).
- **F4b** (physicist: exact S1 + S2 shifts at strength 1): 24 held-out ranks -3.36% [-4.29,-2.42] against F4 0.4 -3.54%; not ported.
  The screen four had said -4.36% against -2.33%.
- **E19** depth-dependent selection floors on the S37 card, 6 fresh ranks (`e19_depth/`, `e19_depth_provisioning.py`; cost proxies = response
  rows x alive columns x bank columns for the read, 1024 x alive x young columns for the push): response floor 0.005 at layers 1-6:
  error +0.2%, read +0.9%; 0.02 at layers 7-10: error +2.9%, read -3.0%; 0.02 at layers 11-14: +1.1%, -3.3%; 0.005 at layers 11-14:
  -0.1%, +2.6%; birth floor 0.005 at layers 0-5: -0.3%, read and push +0.9%; birth floor 0.02 at layers 10-14: +1.2%, -0.5 / -1.1%.
  Every direction is neutral or worse: the uniform 0.01 floor is at its optimum at every depth.
- **E20** collective per-unit kurtosis term k4_coll(i) = 6 sum_r lam_r c_r pp_ir^2 (free): adds 0.03 (layer 9) and 0.003 (layer 13) of
  explained weighted residual on top of K4L's hub terms. Not worth adding.
- **Per-layer injections** (analyst A3 table, four screen networks, units 1e-9 of mean error per layer at depth): from the variance
  error 2.0, from the skew error 1.2, from the kurtosis error 1.3, the three together 3.8 (they partly cancel); carried in 11.5 of 15.0.
- **Trainer**: run M card certified (forward parity with my arms to the digit); its 3-worker pool was refused by the trainer's permission
  system ("interference with other workloads"); told to launch 2 workers x 4 threads (the pool that ran earlier), else 1 worker.
- **Release lane's proposal for the designated file** (TEAM_LOG 03:50Z): the best graded file that lost no network, has >= 0.095 s residual
  room and >= 35 s wall room on the platform's own telemetry, and no lever judged only on its tuning networks: today S37.

## 21. 2026-10-02 04:00-05:40 UTC: after S37. Dials, learned parts, row cuts; run M launched

- **Trainer's finding (04:06Z)**: every refit since the star-Nystrom basis back-propagated with the rank-32 star basis held constant; for
  dials that act through the carried b (bscale, b source, X0, X1, s_carry, F4 age 1, k22) that gradient misses 60-85% of the derivative and
  can have the wrong sign; for the learned arrays both rules agree (cosine 0.9999; birth-coefficient arrays 0.91 / 0.53). Run M uses the
  differentiated basis (same cost).
- **Yet the dials sit at their optimum on fresh ranks**: exact gradients on 16 newer training networks: only X1 beyond 2 standard errors
  (t -2.3) among the 12 dials; forward scans on held-out ranks: bscale +0.01 / +0.02 / +0.03: +0.48 / +1.06 / +1.81% (worse; down is worse
  too); isotropic b-source 0.85 with F4 on: +0.23% (se 0.36); K4L strength 0.75: -0.34% (se 0.16; the physicist's 24-rank run: F4 0.58 +
  K4L 0.73 = -5.58% against -5.28% modelled for 0.6 / 1.0). The trainer's table also marks the covariance-output knob k21n (t +5.1:
  more negative, first order -0.57% per -0.02) and both K4L strengths (down). Network 35 alone had said bscale up -1.0%: wrong.
- **Learned parts on fresh ranks** (`e23_edge/`, `e24_fresh24_shrink/`, `e25_fresh6.log`): mean correction x 0.9: -0.48% (se 0.17);
  x 0.8: -0.84% [-1.50,-0.16] (10 ranks); x 0.7: -0.60%. Its slope on the residual it should remove is 0.85-0.98 at layers 5-14, 0.82 at
  layer 15, cosine 0.17-0.32 (it explains 3-10% of the per-layer residual energy). Learned covariance edge off: +1.00% [-0.43,+2.45]
  for 1.2% of the bill; at half strength -0.08%.
- **Native row cuts** (engineer): the native file is +1.20% (se 0.26) above the torch card over 40 validation networks; cause: the cut of
  young columns on dead rows (alpha < -2.5, layers 7-14), +2.1 / +0.8% on networks 35 / 100; on 3 fresh ranks the band between 2.5 and
  3.0 costs 0.63% of error and saves 0.78% of bill: break-even. No change.
- **Request levers** (engineer, release): L3 three-request operand loads: about -1,260 requests per network on S37 for +0.05% bill,
  byte-identical; release's plan items add up to under 41,500 requests.
- **Leaders** (analyst A14): marius_binner's gains are structural (18 of 27 steps moved flops), no error gain for 21 h, residual 0.363 /
  0.381 s against the 0.4 limit and two entries lost networks to it; J2W lost networks in 8 of 18 entries, wall 97 / 108 s; Luna idle 37 h.
  Generated against public networks: equal means, 1.6x wider spread on generated (cause not found, read as sampling at 33 seeds).
- **Machine**: load average reached 52 with six lanes' screens; nobody's job moved. Rule now: the trainer's two-worker pool has the box,
  every other lane one process of 2 threads, nothing started above a 1-minute load of 12. Agents do not read TEAM_LOG or messages while
  they wait on their own jobs: a pause has to be done by signal and logged. The trainer's permission system refused a 3 x 4 pool
  ("interference with other workloads"); run M was launched at 2 x 4 at 05:33Z after the box was cleared.

## 22. 2026-10-02 06:10-07:10 UTC: what is left after S37 (lead's re-check), candidate 2, plan for S38

**Podium arithmetic (analyst A16, public record of 1007 runs).** Probability that a fresh 100-network run loses a network:
marius_binner 13%, J2W 13%, Luna 5%, we (S37) 2%. With current bests our expected place is 4.1 (podium 23%). Our own score
moves it most: podium 49% at -5% against S37, 78% at -6%, 95% at -7% (1.814e-9). Luna (3rd) is at 1.8437e-9 = -5.5%.

[line removed for publication: competitor per-network telemetry]
suliman_tadros 2.73, marius_binner 2.65, J2W 2.38. J2W's layers 0-14 are 7% WORSE than ours and its final layer 10% better.
Our own profile has no special defect at the last step: on 11 held-out ranks the S37 card's error ratio layer 15 / layer 14 is
1.15, as between earlier layers (1.09-1.19); mean per-layer MSE x1e9: 0.68 0.63 0.80 1.24 1.88 2.67 3.62 4.68 5.74 7.04 8.50
9.78 11.55 13.13 14.33 16.35.

**Bill structure (engineer's census of the S35 lineage, network 100).** Bank push 43.7%, young plane read 35.2%, covariance
push + star 10.9%, final layer 3.4%, learned edge 1.5%, pair map 1.2%. The young tier holds about 5,900 columns at layer 10
(nearly every alive unit is a birth hub: about 900 pairs per layer, three cohorts young), so pushes and reads are products of
shape n x n x 6n. The newborn cohort's first push is already one column per hub (common part once, directions gathered from
the weights). Re-read of the September record: exact old tier = every-3 joins within noise; value-weighted join rows, cheaper
join sketches, rank cuts with or without refit, per-cohort CP compression: all closed. Nothing structural is left in the bill.

**Closed this morning (held-out ranks, S37 physics, forward only):**
- E26 second-order skew term of the mean formula, (k3^2/72) x sixth derivative (`e16b` arm `e2`): rank 651: +0.8% with the
  learned correction on, -2.2% with it off (layers 12-15 better, 7-9 worse). Real but already absorbed; only as a trained strength.
- E27 (`e27_birth_dials.sh`, `e27_summary.py`, six ranks): s_new -0.01: +2.4% (6 of 6 worse), -0.02: +8.7%; X1 +0.05: -0.05%
  (se 0.25), +0.10: +0.07%; birth coefficient d (1 + x radial/3): x 0.5: +0.10% (se 0.31), x 0: +0.84%. The first four ranks
  had read -0.4% for two of these arms: six ranks are the minimum for a dial.
- Re-derived and dropped without a run (already closed in memory): gain frame, Monte Carlo hybrids of any kind, a fourth-order
  bank (one more bank of bill), exchange terms at late layers only, exact all-hub births at the last layer (exact skew at layer
  15 alone is only -1.6%), rank extrapolation, pair structure in the read (two columns per pair either way).

**Trainer (07:00Z): the gain of a refit on this card is in the dials, found by line search.**
- Learned arrays: one large step along the mean of 36 exact network gradients is +0.21% (se 0.12): no room (optimum -0.02%).
- Dial-only candidate 2 = learned mean correction x 0.8, K4L 0.75 / 0.75, k21n -0.048, b-source 0.965, bscale 1.03414
  (was 1.03834), X1 0.59158 (was 0.55658), F4 strength 0.656 (age 1) / 0.6 (age 2): -1.44% on the 50 held-out ranks
  (se 0.26, 40 of 50), all layers -1.15%. The step from candidate 1 (-0.98%) was chosen on 16 other held-out ranks.
  File `trainer/candidates/runM_dials2/candidate.py` (sha 716f66dd) + levers F4v4t0.656,0.6 and K4Lv3c0.75,0.75.
- Line searches along the exact dial gradient are about ten times faster than the optimiser; run M as an optimiser is over,
  its pool is the trainer's evaluation pool.

**Physicist (06:15Z):** exact-term mixture (rule 0.378 + S1 0.858 / S2 0.48, birth constants) + K4L 0.71 against F4 0.58 +
K4L 0.73 on ranks 25-48: -0.46% [-0.92,+0.04]; 48 ranks: -0.48% [-0.83,-0.12], 34 of 48; bill +0.06% as one quadratic form.

**Plan.** S38 = candidate 2 + S37 stack with the per-age levers + edge cut (EcutS) + operand loads (L3, L3o): expected -2.1%
(1.951e-9 -> about 1.91e-9). Engineer builds and runs fresh list a; lead runs `release/lead_chain.sh <file> <tag>` (stages A, C,
D) in a quiet window and fires. Then F4c port + candidate 3 (line search at candidate 2 with the e2 and mixture strengths).

## 23. 2026-10-02 07:10-08:35 UTC: S38 qualified, fired, graded 1.907e-9; three more closed screens; the second-chaos question

**S38 = #333481, graded 08:33Z: 1.907e-9** (S37 1.951e-9: -2.26%). Final-layer MSE 1.612e-8 (-1.65%), all-layer 6.377e-9,
budget share 0.11830 (-0.60%), 100 of 100 networks, platform wall 77.1 / 88.3 s, residual 0.294 / 0.332 s. File: engineer's
stack on the trainer's candidate 2 (sha 7c1a67f4, archive 88f51621). Evidence before the fire: torch dials -1.44% on 50
held-out ranks; native on 10 held-out ranks -2.03% [-3.12,-0.99] error, -0.624% bill. Chain (`release/lead_chain.sh`, 32 minutes
in a quiet window): A PASS; C PASS with same-window legs (projected wall 74.9 / 86.3 s, residual 0.295 / 0.328 s, bill
0.97871 x S35); D PASS (handles 57,578; group RSS 6.81 GB, which the release lane traces to the quiet box, footprint 2.20 GB).

**Closed (held-out ranks, forward only):**
- E28 clamps of the birth pair geometry: the discriminant floor binds on 13-22% of the hubs at layers 6-14 (always-on units),
  the mismatch left is at most 1.3% of the summed |innovation| and 0.1% of its energy.
- E29 kurtosis term with pairs of ages 3-4 at reduced weight: final layer only 0.5 / 1.0: -0.04% / +0.04%; all layers 0.3 /
  0.6: -0.06% / +0.04% (six ranks).
- E30 third-to-fourth conversion carried as a second weight per bank column (`e16b` arm `cv`, rank 651): accumulated +3.8% at
  strength 1, +1.5% at 0.5; the one-layer source alone -0.1%. Right at early layers (layer 6 -1.5..-1.8%), wrong at depth.

**The second-chaos question (asked of the physicist 08:25Z).** Weights are iid Gaussian (analyst's loader). For unit i of the
next layer K4post(w_i^4) = constant + (12/n)(w_i' N w_i - (2/n) tr N) + fourth chaos, N_ab = sum_c kappa4(x_a, x_b, x_c, x_c)
(how the covariance of a pair moves with the layer's squared norm). In pre-activation space Ntilde_ij = Cov(z~_i z~_j, |z~|^2)
- 2 (C^2)_ij, and k4_i = a v_i^2 + (6/n) Ntilde_ii + ..., K31_ij = (3/n) Ntilde_ij + ..., feed kappa4(i,j,k,k) = Ntilde_ij / n
+ .... Collective content is second chaos (larger than the fourth by order n), hub-local content is both. The third-order
analogue already exists in the card: the trace vector t_a = sum_c K3[a,c,c] = Cov(x_a, squared norm) is the b carrier. If the
Monte Carlo shows that Ntilde explains most of the unit-specific kurtosis and of K31, the missing carrier is a matrix B with
K4 = Sym(C x B) (about +2-5 products per layer, +10-25% bill); if not, the content is fourth chaos and no matrix carrier exists.

**Trainer's per-layer signal (07:12Z):** 22 of 300 per-layer dial scalars beyond 3 standard errors on 38 networks: s_carry up at
layer 14 and down at layers 1-4, s_new up at 13, bscale down at 7-14 and up at 0-3, K31 row factor and k31 knob up at 14,
final-layer k4 down, k22 down at 14: end effects. Candidate 3 = per-layer tables by line search.

## 24. 2026-10-02 08:35-10:15 UTC: the second-chaos matrix of the fourth cumulant and a bill-free carrier for it (NCAR)

**Chain of measurements (network 35 unless said; scripts e31-e36, ncar*.py; physicist's T19 series).**
1. E31 (CPU Monte Carlo, per unit): Q_i = sum_k kappa4(z_i, z_i, z_k, z_k) explains 36-46% of the card's per-unit kurtosis residual at
   layers 5-15 with slope 5.0-5.7 / n (chaos expansion for Gaussian weights: 6 / n); it is not a function of free per-unit quantities.
2. E32 (full matrix Ntilde at layers 9, 12, 13): tr C x C and C^2 with free weights hold 78-80% (C^2 weight 23-40 times the card's ratio);
   mean forms bring it to 85-86%; the network-specific rest is low rank (rank 32: 78-85%, rank 64: 91-94%) and follows a one-layer
   transport law (slope 0.77 of W' G N G W; 2 mean(g - m d) = 0.86).
3. Physicist's oracle (true network-specific part, all layers): as an extra K31 plane (3/n) R_ij: -8.0% (35) and -15.6% (held-out 651),
   rank 32: -6.1% / -15.0%; as per-unit kurtosis: worse (+3.2% on 35; my free C^2 version E33: +3.3% on four held-out ranks); as feed: nothing.
4. E32c: in the eigenbasis of C the matrix is F(C) (82-84%) + one-sided block of the top eigenvectors (top 8: 93-94%): the network-specific
   part is 70% "collective mode x per-unit partner vector". E32d: given truth one layer back the partner vectors of modes 2-8 are 81-97%
   predictable, mode 1 (the mean direction) 55-63%.
5. E35 (matrix before the ReLU, after it, mixed, and at the next layer): pre pairs with the post norm = gammabar x pre pairs with the pre
   norm (98-99.6%); the push is exact; the ReLU step is the leak. E35c: the ReLU step is the Gaussian linear response to the collective
   tilt: gate term + Gaussian births + own (3,1) + mean-tilt row form d t g C + VARIANCE-tilt row form -(d alpha / 2 sigma) P_aa g C:
   84-85% (80-82% with every coefficient at 1). Physicist (09:50Z) adds homogeneity (the pure-gain part of the tilt maps exactly on the
   post covariance) and reaches 88-95% per step with nothing fitted.
6. NCAR (`ncar.py`, law 2; spec `NCAR_SPEC.md`): the partner vectors carried free-running with matrix-vector products. Open loop on 35:
   cosine with truth for modes 2-8: 0.86-0.91 (layer 2), 0.46-0.74 (layer 13); energy held 0.79 -> 0.40. Closed loop as an extra K31
   plane: network 35 -0.75% (K 8), held-out 651 -3.55%; first three held-out ranks pooled: K 8 -2.1% (modes 2-8) / -2.9% (modes 1-8),
   K 16 -3.6% / -4.2%. Screens E37 (12 held-out ranks) and E39 (design variants) running at the time of writing.

**Closed on the way:** E30 conversion carried on bank columns (wrong at depth); E33 per-unit C^2 kurtosis (+3.3%); E34 per-column
fourth-order weights from a column's own history (nothing beyond rule A); E35b naive pair-map derivative with the own-unit rows left in
(my error, corrected in E35c); E38 mode 1 from the bank's trace vector (true partner has cosine 0.6-0.68 with it, but no gain in closed loop).

**Other lanes:** trainer's candidate 3 (joint solve over 12 single changes on the exact-term card): -0.87% against S38's dials on the 50
held-out ranks (se 0.19); engineer: F4c at the physicist's literal strengths gives nothing on S38's dials natively (+0.18%, bill +0.13%);
star with exact row sums -0.24 / -0.52% on two networks for +0.036% bill, lower star ranks being priced; release: fast path for
literal-only candidates works, memory gate moved to footprint, wall projection corrected for page-fault work.

## 25. S39 graded 1.8470e-9; the carrier splits into a free bank channel and a structured low-rank state (2026-10-02 11:40-12:30 UTC)

- **S39 = #333528: 1.8470e-9** (S38 1.9070e-9, -3.15%): final-layer MSE 1.5970e-8 (-0.93%), budget share 0.115675 (-2.22%), 100/100
  networks, wall 77.3 / 87.6 s, residual 0.302 / 0.347 s. Fourth, 0.18% behind Luna (1.8437e-9). The ten-rank native estimate of the
  error change had been -0.04% (se 0.39): ten ranks size an error change to about +-1% only.
- **Where the sketched carrier lost its value (held-out 651, base 16.008e-9; `e45_651/rank0651.log`, `rank0651_e46.log`).** Full-matrix
  complete law -15.1%; lean law with an ideal rank-32 cut -10.6%; my sketch (`ncar4.py`) at rank 32 with 8 eigenvector + 16 Gaussian
  probes -2.0%; with 40 random probes -6.9%; with the probes taken from the carried directions (8 eigenvectors + the 32 transported
  columns, option `ncpr=1`) -8.0%; rank 16 with 24 such probes -4.0%; the certified law (row dial 0) -7.7% against -8.0%. So the probe
  count was the weak point, and 40 two-pass probes cost about 15% of the bill natively: the sketch is not a native form.
- **`ncar5.py`: a structured state without probes.** The fresh sources of the lean certified law are covariance-shaped; with the layer's
  covariance replaced by its top-K eigenpairs inside the births and the row forms, and every diagonal sandwich, the gain term and the
  conversion dropped, the fresh part is an explicit small matrix on the columns [W'(g o Z) | W'(g o U) | W'(rv o U)]. On 651
  (`e45_651/rank0651_e47.log`): rank 32, K 16: -3.12%; rank 16, K 8: -1.93%; without the Gaussian births -3.39%.
- **The bank channel (physicist's `ncar_bank.py`: a second weight per bank column) on 651 on my base: -4.64%; both together -7.55%**
  (they add). The bank channel is free to carry and dear to read as a separate plane (+35% bill); the physicist's fold into the K21 read
  (`ncar_fold.py`) is the native form and the S40 question.
- Native count of `ncar5` per layer: r + 2K products with W (rank 32, K 16: 0.70% of bill), the eigenpairs (0.7-1.4%), the plane
  (0.2-0.4%): near break-even on 651, less pooled; kept as the second step after the fold.

## 26. The whole lean law on the bank's own column pairs (`ncar6.py`): the S40 file (2026-10-02 12:25-13:50 UTC)

- **Observation.** The card's newborn bank columns of a birth hub h are `p+- = s_h + t+- e_h`, `s_h = g o Cpre[:, h]` (zero own entry; plus the
  birth-direction changes). The physicist's lean law for the second-chaos matrix has, in post space, the fresh sources
  `sum_h S2_h c_h c_h'` (births), `sum_h rv_h (e_h c_h' + c_h e_h')` (row forms), `sum_h D_h e_h e_h'` (diagonal) with the same
  `c_h = g o Co[:, h]`. So every hub is a symmetric 2 x 2 form `[[S2, rv], [rv, D]]` on its own pair, and the bank's own transport carries
  it: the physicist's second weight per column plus ONE cross weight per pair. Formulas and the native op list: `NCAR6_SPEC.md`.
- **Held-out 651** (base F4 0.6 + K4L 0.75, 16.008e-9; `e45_651/rank0651_e48.log`, `_e49.log`): bank channel only -4.64% (equal to the
  physicist's module to the digit); + row forms -12.44%; + diagonal -12.36%; + births -14.43%; without the isotropic-rest sandwich -14.90%
  (full-matrix complete law: -15.1%). Fold inside the bank read: four exact forms -13.84%, exact multiple of C -12.63%, strength 1.25 -14.65%.
- **Seven networks** (35 287 328 378 424 641 651; `python e51_summary.py e51_pairlaw`): reference plane -7.49%, fold with four exact forms
  -7.46%, fold with the exact multiple of C -5.78%, fold with the native table (RHO1 x diagonal regression) -5.60%. The 1.7 points between
  one and four forms sit on 287 and 641.
- **Where those 1.7 points are** (`e53_forms/`, `e54_center/`, `e55_ptop/`, `e56_bulk/`, `e57_pdiag/`, `e58_pmu/`, `e56_spectrum.py`): the C^2 form,
  not the mean forms (exact two / three forms -1.8 / -0.8% on 287 / 641); not the births (centring them +1.2%, dropping them +3.2% on 287).
  In the eigenbasis of the covariance the carried matrix is 80% diagonal, and the multiple of C it carries falls from mid-spectrum to the top
  modes; the four-form fit's multiple of C is 0.93-0.97 x the diagonal regression's. Removing 0.95 x that coefficient x C and then the
  Rayleigh quotients of the rest on the top-K eigenvectors gives -3.2 / -4.5 / -5.2 / -5.5% on 287 at K 4 / 8 / 16 / 32 (641: -0.3 / -1.2 /
  -3.1 / -3.4; 651: -12.5 / -12.6 / -12.8 / -13.4); the mean direction or one or two eigenvectors are not enough (-2.9 to -3.4% on 287 and
  WORSE than the table on 651: -10.1 to -11.5% against -12.4%). Natively K 16 costs about 1.6% of bill for about 1.5 points: not in S40.
- **Team, same hour.** Trainer, first 25 judging ranks on candidate 3: -6.07% (se 0.76, 24 of 25); three forms (table for C, diagonal
  coefficients for the mean forms) -6.58%. Engineer, NATIVE lever NP on S39, ten fresh networks: -7.50% [-9.45, -5.49], 10 of 10, bill
  +1.17% (parity version; lean target +0.9% and +900 calls), native = torch network by network. Physicist: reading audit passes; the same
  matrix's diagonal in the K22 plane adds 0.7 / 1.75 points on two networks (held-out check queued).
- **Machine rule changed 13:16Z.** The load average read 51-90 with the machine 55% idle (browser, another session's job); every lane's
  screens were waiting for nothing. CPU gate is now `idle_gate.sh 30` (idle share over 2 s); memory rules unchanged.
- **Decision 13:40Z.** S40 = S39 + NP lean (three forms, the card's beta in the carried weights, sources and fold at strength 1), fired when it
  qualifies; dials, K22 term and any top-mode repair go into S41.

## 27. S40 graded 1.7590e-9 (third place); the fourth-chaos plane K31L4 for S41 (2026-10-02 13:50-15:30 UTC)

- **S40 = #333559: 1.7590e-9** (S39 1.8470e-9, -4.76%): final-layer MSE 1.5070e-8 (-5.64%), all layers 5.928e-9 (-6.6%), budget share
  0.116720 (+0.90%), 100/100 networks, wall 79.8 / 91.2 s, residual 0.311 / 0.365 s. Third place (Luna 1.8437e-9 behind; marius_binner
  1.6407e-9 and J2W 1.6161e-9 ahead). File: S39 + engineer's lever NPf3bv4 (pair-carried law, three forms by the table and the diagonal
  regression, the card's beta in the carried weights); chain 14:24-14:52Z, fired 14:52:37Z.
- Evidence chain before the fire: torch 50 judging ranks -5.80% (47 of 50; trainer); native parity version on ten fresh networks -7.50%,
  the fired file on the same ten -7.93% (engineer); native = torch network by network; physicist's audit (the pair matrix equals the
  full-matrix lean law without its gain term to cosine 0.988-1.000, and the gain term is worth nothing in closed loop).
- **Forms.** All four form coefficients come from a regression of diag B on (tr C v, 1, mu^2, 2 mu (C mu), (C^2)_ii) and reproduce the
  exact four-form fit (`e59_const/`); the matrix C^2 is the cost: 16 columns of a warm-started subspace iteration do it on 287 / 641 / 651
  but lose on 35 (-3.8% against -6.2% with no C^2 removal; 32 columns needed), so about 1.3% of bill for 1.2 points: closed for now
  (`e60_c2sub/`, `e63_ceiling/`). The physicist's constant regressor replaces the 14-literal table (-5.81% against -5.60% on seven networks).
- **K31L4 (`k31l.py`).** K4L's local quartic form of the young pairs is used on the diagonal only; the pair law carries its second-chaos
  projection in the K31 plane; the fourth-chaos rest of Q(i,i,i,j) was in no plane. Ages 1-2, strength 1, nothing removed, folded into
  the same read: seven networks -7.60% against -6.13% for the law alone (6 of 7 better; `e62_k31l4/`); with exact four forms -8.81%
  against -7.46% (`e63_ceiling/`). On 651: age 1 only is worse (+0.96%), ages up to 3 / 4 give less (-1.2 / -1.5%), strength 0.5 / 1.5 less.

## 28. S41 fired; anatomy of the true (3,1) plane; where the cheap physics ends (2026-10-02 15:30-17:10 UTC)

- **S41 = #333587, fired 16:51Z** (grading when written): S40 + K31L4 (ages 1-2) + the trainer's first joint dial solve of the law
  (fold strength 1.83 / 1.40 / 1.18 by layer band, factor 1.272 on the removed multiple of C, births 1.142, rows 0.697, diag 0.791,
  c0 0.6222, k31 knob -0.1659) + request diet (HV, Rlvfs_limit64, Rl3_bind128). File = `release/variants/s41hv_33bac9db_lvfs64_l3b128/
  candidate.py` (sha 169fd371), archive sha 58ddd119. Qualified WITHOUT a quiet window (release lane's stage C and D under load, instruction
  counts; stage D's swap line not judged). Evidence: native ten fresh networks -3.26% error, +0.30% bill against S40.
- **Second dial solve** (trainer): -0.84% against the S41 stack without an age-3 path (-1.03% with it); exported as
  `trainer/candidates/law_dials_d7_ages12` and `law_dials_d7`: S42 material, not yet worth a submission by itself.
- **What the true (3,1) plane is made of** (`e66_k31_truth.py`, network 35, 8M truth planes): law 12-16% of its energy, + K31L4 ages 1-2:
  19-23% (14% at layer 14), chains +1-2, MIXED terms (third cumulant x curvature x one Gaussian edge; `k31diag.py`, `k31m.py`) +17-19
  points, all at coefficient 1; 55-65% unexplained. Physicist's restart Monte Carlo: the missing content is made by all earlier layers about
  evenly; two thirds is the previous layer's whole fourth cumulant carried through the gates, 19-32% the conversion of the third cumulant.
- **Closed loop and cost**: mixed terms -2.0 / -3.2% (35 / 651) for a product over hubs x all bank columns (+27% bill); chains -0.8% for an
  n^3 product (+4-5%); K21 gain on the kink rows is catastrophic (+5%: error x 2.6; `e68_k21gain/`); the matrix's diagonal in the per-unit k4
  loses at every strength, also with the learned correction off (`e69_learned_off/`). The (3,1) slot is closed on cost beyond the law +
  K31L4; a bill of +25% for -20% error would only move us along the frontier (it is roughly marius_binner's position).
- **Budget after the law** (analyst A20): the variance error the mean sees at layers 12-15 is plane error: K21 on kink rows half, K31 a
  third, K22 5%. The learned mean correction is now worth 3-8% (it was 4-11% without the law): a refit has 1-3% to gain.
- **Plan**: S42 = second dials + byte-identical diet; round 2 of the card's dials; the learned-array step test and, if it shows room, a
  refit on a physics base that includes the terms that are right but lose under the present learned parts (second-order terms, b-form).

## 29. S41 graded 1.7110e-9; new INPUTS of the learned head are the next lever; the refit base is weak; two closures (2026-10-02 17:10-19:35 UTC)

- **S41 graded 18:22Z: 1.7110e-9, third** (final-layer error 1.4620e-8, share 0.117035, wall 83.6 / 90.2 s, residual 0.318 / 0.354 s).
  Board 19:11Z: J2W 1.5883e-9 (error 1.4567e-8 = ours, share 0.1091, residual 0.369 / 0.398 s, wall 105 / 119 s), marius_binner 1.6407e-9
  (error 1.1405e-8, share 0.1438, residual 0.368 / 0.394), we, Luna 1.8430e-9, suliman_tadros 1.9339e-9. Analyst: J2W's whole lead is its
  bill (ours x 0.98 minus 13-18 Gflop on every network = one full product per layer); chance that a fresh 100-network run loses a network:
  J2W 56%, marius_binner 34%, suliman_tadros 46%, Luna 4%, S41 2%. Release lane: at the leaders' risk level our kernels would save under
  0.5% of bill, so their lower bill is structure.
- **The head-input law** (trainer 18:41Z-19:26Z): the unit-specific diagonal of the law's matrix, (B_ii - fit) / (tr C v_i) on the read's
  response rows, as a 31st input of the learned mean correction: 105 of its 128 zero weights beyond 3 standard errors (mean t^2 55; existing
  columns 1.4-11); one line-search step: -1.79% on the 25 judging ranks (23 of 25), two steps -2.14% (24 of 25). Physical reading: 6 x_i is
  the unit-specific kurtosis the second-chaos closure implies = how strongly the unit's variance follows the collective scale. Direct physics
  uses of the same number lose (learned on and off); the head shrinks and calibrates it. Native: free (the fold holds the diagonal; one 5 x 5
  solve, 21 requests a layer). **S42 = S41 + the column**: native ten fresh networks -1.55% (8 of 10 lower, two level), bill -0.03%; stage C
  PASS (43,470 / 42,573 requests, wall 81.8 / 88.2 s, residual 0.320 / 0.356 s); stage A PASS (one new signature read: ij,c->ijc, an outer
  product of a kept block with 16 channel constants in the edge cut v2); stage D running when written.
- More candidate inputs are in the trainer's card at zero weights (parts of the diagonal, K31L4's diagonal, W' applied to the law's per-unit
  sources, the parts of per-unit k3 and k4); the existing kurtosis column is off its optimum (64 of 128 weights beyond 3 se).
- **Refit base, interim** (physicist, S41 torch card, 651 / 378 / 424, learned correction off): the ladder m + t + f + bsk reads -4.69 / -0.04 /
  +0.67%; with the learned parts on only the skew-corrected birth is level or better (-1.00 / +0.01 / -1.18%). The base terms do not add and
  are not robust across networks: the refit on a physics base is a weak lever; new information (inputs, planes) is what has paid today.
- **Closed**: (i) the newborn bank columns taken from the post covariance, to share one full product per layer with the covariance sandwich
  (4.15% of bill): +270..+330% error on 651; the two column sets differ by 6-21% (`e74_sharedcol/`). (ii) Conversions of the third cumulant
  into the fourth (every form needs a product of hubs x all bank columns; sparse, sketched and newest-cohort versions all cost more bill than
  they return). (iii) The law's matrix as a feed of the third cumulant and as per-unit k4 (physicist T20). The second-chaos sector is
  saturated: the pair-carried law holds about 90% of what the exact matrix gives on held-out networks (-5.8..-6.1% against -6.5%).
