# fable_oldtier_economics_v1 — session notes (started 2026-09-22 ~13:15 PDT)

Goal (Matt's /goal): re-examine phase-2 work, bank 3e-9 / 2e-9 / ~1e-9 adjusted. Official bank: #331824 adjusted 3.5071e-9
(raw 1.9006e-8, util ~0.1845, wall worst 110.2 s of 120, residual worst 0.385 of 0.4 s). Board 2026-09-22: J2W 2.1e-9
(raw 1.64e-8, util 0.131); marius 2.4e-9 (1.68e-8, 0.146); suliman 2.4e-9 (1.83e-8, 0.132); a_s6 2.5e-9 (2.00e-8, 0.126);
Puffi/reds 2.8e-9; oqaris/ben3 3.1e-9; Camaro 3.3e-9; us 3.5e-9 (rank 10).
Arithmetic: 3e-9 = -14.5% bill at fixed predictions (or -21% bill at +8% raw, etc.); the top-4 sit at OUR raw with ~30% less bill.

## Public record (phase 2 only; Matt: [instruction removed])
- 504aldo t/18218 (MIT code in references_public/whest-p2-cumulant-k3): factorized K=3 chain, raw 2.13e-8 at 0.2526.
  Bill = 95% third-cumulant sources: young dense tier 116 u, old shared-basis tier 107 u (1024 u = B). "Leaders' multiplier
  = our bill WITHOUT the old tier to three digits." Dead ends: dropping old sources 6x worse; memoryless slice closure 45x;
  CP column caps (4n -> D21 err 0.39); Tucker core r^3/n^2 units; adjoint no cost drop; learned corrections ~1.45x at best.
  eps law: extra MSE ~ 4.2e-6 x eps^2 (eps = rel rms error of D21) -> D21 must be ~2% right.
- Organizers' paper 2605.05179 (references_public/writeups): factorized K=3 = 30 n^3 L^2 + ... FLOPs.

## Our card (331824) measured today (r35 = rank-35 research net, 1e9 truth; r100)
- bill census (callers): read kernel 26%, young push 17%, star push 12%, old tier (push/reconstruct/inner/join + listcomp) ~24%,
  cov push 8%, port body 4%, rest ~5%. r35 util 0.1883, MSE 1.648e-8; r100 util 0.1846, MSE 1.680e-8.
- OLD-TIER PRICE BY AGE (oldtier_price.py; weights of cohorts aged >= A zeroed after the push; bill unchanged):
  r35: A=16 1.648e-8 | A=11 2.74e-8 (+66%) | A=9 6.48e-8 (3.9x) | A=7 1.34e-7 (8.1x) | A=5 3.00e-7 (18x)
  r100: 1.680e-8 | 3.10e-8 (+85%) | 7.22e-8 (4.3x) | 1.73e-7 (10x) | 4.87e-7 (29x)
  -> old content is essential at EVERY age; effects add coherently (RMS grows ~linearly with cohort-layers removed).
- BANK STRUCTURE (dump_bank.py + analyze_bank.py, r35): old plane norm / total plane = 25% (L6), 45% (L9), 57% (L12), 60% (L14);
  cos(old, young) 0.3-0.4; old plane as a matrix: energy 73% rank-1, 89% rank-16, 98% rank-64, 99.99% rank-256 (L14).
  Per-cohort columns 1344-2048 (births fall 1024 -> 672 with depth); cohort column rank(99.9%) 150-190 at L14.
  Symmetric CP-ALS of each cohort's tensor (own frame): rel plane error at K'=1/4/16/64/256 = 0.93-0.97 / 0.81-0.93 /
  0.62-0.85 / 0.43-0.70 / 0.22-0.51 -> CP compression of cohorts is DEAD (matches 504aldo F76).
  Cohort plane cosines 0.1-0.56.

## Consumer + pruning + regeneration (r35, 2026-09-22 ~14:00-14:30)
- consumer_ablate.py: old PLANE read zeroed (diag/radial intact) -> 1.79e-7 (11x); old DIAGONAL zeroed (plane intact) -> 2.28e-7 (14x).
  Both consumers essential; no consumer-side cut.
- birth_prune.py (importance = born-tensor Frobenius norm birth*sqrt(3|b|^4+3u^2|b|^2+(t^2+ts+s^2)^2)):
  keep 1.0: util .1883 / 1.648e-8 | keep .75: util .1589 / 3.25e-8 (2.0x) | keep .5: util .1236 / 1.23e-7 (7.5x).
  (v1 importance birth*|b|^2 was slightly worse: .75 -> 3.76e-8.) The bill scales with columns as hoped (-16% at .75, -34% at .5).
- regen_regress.py (in-sample, 14 live matrices incl. the memoryless transport of the previous old plane): old plane R^2
  all = 0.53 (L6), 0.68 (L9), 0.69 (L12), 0.68 (L14); without the memoryless feature 0.27-0.60. Need >= 0.99 -> regeneration DEAD.
- learned-basis fit (previous session's live worker): pass-2 vs pass-1 same-network ratio 0.996 at 289/320 graphs (needs 0.94
  to pay its +6.2% cost) -> negative; not a bank candidate.
OPEN: is the 2x from pruning 25% information loss or closure co-adaptation? -> prune_info_loss.py (plane/diag rel error on the
full dump when the same columns are zeroed) vs the eps law (D21 2% -> +8% raw).

## Oracles / young rank / FLMS (r35, ~14:40-15:00)
- oracle_planes.py on 331824 at L12-15 (8M MPS truth planes): truth K21 -> 4.03e-8 (2.4x WORSE: the learned closure is tuned to the
  card's own K21); truth K22 (incl. per-unit k4) -> 1.507e-8 (-8.6%); both -> 3.77e-8.
- oracle_diag.py on 331824 at L12-15 (8M per-unit truths): true k3 diag -> 1.596e-8 (-3.2%); true per-unit k4 -> 1.536e-8 (-6.8%);
  both -> 1.456e-8 (-11.7%). (c2 had -5/-10/-17%; the edges ate part of it.)
- young_rank.py: young cohorts' column energy: age 4 -> 99.99% at rank 384-512; age 3 -> 512; age 2 -> 640; age 1 -> 768 (99.99%
  is the working criterion: rank 256 at age 4 = 99.8-99.96% energy yet +22-33% raw). Earlier confinement saves <= ~5% after Winograd.
- flms_switch.py: _FULL_LOWER_MOMENTS=False (alive-compact cov/star pushes): util -3% but raw +6% (r35) / +10% (r100) -> worse.
- prune_info_loss.py: zeroing the lowest-importance 25% of births changes the consumed plane by 13-16% and the diagonal by 13-15%
  at every layer (50%: 37-40%): importance is FLAT; the 2x raw loss is real information loss, not just closure breaking.
CONCLUSION so far: every local carry-less dial on this card is a cliff; the exact star-atom carrier is at its cost floor
(K columns x rank r x (n+R) per layer); accuracy headroom from truth cumulant supplies at L12-15 is ~7-12% before refit.
- star_ablate.py: star_pre zeroed all layers -> 2.505e-8 (+52%); layers >= 8 only -> 2.054e-8 (+25%); layers < 8 only -> 2.099e-8 (+27%).
  The star (12% of the bill) carries real content; dropping it is -12% bill for +52% raw.
- Torch port fidelity (learnable_joint_basis_training_qualification_v1 call_00 vs native r35): L15 rms diff 4.2e-6 (error scale
  1.3e-4); torch MSE 1.6491e-8 vs native 1.6480e-8; forward ~9 s single-thread CPU, fwd+bwd ~23 s. Valid refit surrogate.
- fable_refit_v1/refit_probe.py (research copy of the torch bank model, asserts relaxed): r35 forward 3.5 s at 8 threads, fidelity
  4.2e-6 RMS vs native; fwd+bwd with cstar(15x4)+A(64x4) trainable 13.0 s, peak RSS 8.0 GB; gradients finite (|g_c| 1e-9, |g_A| 6e-9).
  => one refit epoch over TRAIN160 ~ 35 min in one process.
- learned-basis fit (learnable_joint_basis_fit_v1) COMPLETED 320/320 (terminal_basis_payload.npz written): same-network pass-2/pass-1
  geo-mean 0.9952 (117/160 wins) vs the ~0.94 needed to offset +27.5 GF (+6.2-6.5%) -> CLOSED NEGATIVE; no native exporter/evaluation
  will be built. No live numerical workers remain from the previous session.

## Unit oracles (oracle_units.py, r35, control 1.648e-8; pair maps AND the final nonlinear_scalar wrapped)
- L15 only: true v -3.8% | true k3+k4 -8.0% | v+k3+k4 -12.7%.  L12-15: v -10.5% | v34 -24.0%.
- L8-15: 34 -24.6% | v -22.1% | v34 -30.1%.  L1-15: 34 -26.3% | v34 -13.9% (early-layer truth v HURTS: co-adaptation).
- final_formula_test.py: nonlinear_scalar with EXACT (8M) inputs reproduces the truth within the 8M inputs' own noise at every layer
  (L15: 8.1e-9 vs noise floor 8.4e-9; Gaussian-only 1.9e-7). The marginal formula is not the bottleneck; its INPUTS (v,k3,k4) are.
- fable_refit_v1/k4_hybrid_diag.py: adding the exact (2,2)+(4)-slice residual of post22 (which the card collapses to 'radial') to the
  next layer's k4 changes NOTHING (k4 rel err 0.14-0.27 -> same; corr +0.003; K22 off-diagonal unchanged; apply-mode final 1.6499e-8
  vs 1.6489e-8). The per-unit k4 structure lives in the (3,1)/(2,1,1)/(1,1,1,1) slices; the Gaussian-born tree diagrams for it
  cost ~4 units/layer (+2%/layer) -> dead by economics. Isotropic k4 stays.
- PLAN (RF-2, refit.py): drop old cohorts by age (A=9: -13% bill; A=7: -24%; A=6: -35%) and JOINTLY REFIT P (GNN 45,200) + A + cstar
  + edges on TRAIN160 (13 s/net fwd+bwd, 35 min/epoch); VALID40 forward after each epoch. Pass bar: valid raw <= +5% at A=9 (-> ~3.2e-9),
  or <= +18% at A=6 (-> ~2.9e-9). The refit tests whether the old content's EFFECT is learnable by the closure, not the tensor.

## Refit line (fable_refit_v1, 2026-09-22 ~14:00-15:00 machine time)
- refit.py drop-age proxy calibrated on VALID8 (torch): A=9 -> 7.8e-8 (3.6-4.8x), A=7 -> 2.9e-7, A=6 -> 1.87e-7 vs control 1.89e-8.
- refit_drop9_lr1e3 (all groups, lr 1e-3 cosine, clip 5) DIVERGED: train 7.8e-8 -> 1.7e-6 by update 6. Relaunched at lr 2e-4, clip 1.0.
- build_drop_variant.py: native cand_drop<A>.py (columns really removed; lam/cohorts compacted; core decremented; lazy group dissolves
  when its prefix empties). cand_drop16 = parent bit-exact (r35 1.648017257e-8, util .18829).
- Request census of the bank at the production boundary (run_multinet_rss.sh, under refit load): cold r35 58.4k requests (frames 10.1k,
  kernel 23.8k, body 24.6k), later nets 48-50k; walls 52-62 s contended (quiet ~40 s); server peak RSS 6.25 GB. Platform passed at
  93-110 s. Body requests (~23k) are the cheapest wall lever (fusions), ~1.2 ms each on the platform.
- export_params.py round trip (params_orig.pt -> cand_drop16_reexport.py): r35 MSE 1.648017257e-8 / util .18829 = bit-exact. Deploy path
  (refit .pt -> native candidate) is validated. Native drop7: util .16032 (-14.9%), MSE 1.38e-7 before refit.
- refit_drop9_lr2e4: train running mean 1.33e-7 (after the first step) -> 6.27e-8 by update 10 (pre-refit 7.8e-8, control 1.89e-8).
- refit_drop9_lr2e4 EPOCH 0: VALID40 mean 7.71e-8 -> 4.71e-8 (geo-mean per-net ratio 0.61, range 0.51-0.71) but still 2.57x the
  control (1.89e-8 on the 8 common nets). The drop-9 variant would need <= 2.07e-8 to pay its -8.5% bill. NOT going to pass; the old
  content's effect is not learnable by the closure at the needed precision. Letting epochs 1-2 finish for the record; chained control
  refit_drop0_lr1e4 (no drop, joint retune of all groups, 2 epochs) follows.
- subframe_info_loss.py (r35, static plane proxy): confining cohorts aged >= 9 to a nested sub-frame: rank 256 -> plane rel err
  0.17-0.28%, diag 0.10-0.16% (energy 99.97-99.998%); rank 192 -> 0.5-0.9%; rank 128 -> 1.3-2.5%. Age >= 8: rank 256 -> 0.3-0.5%,
  rank 192 -> 0.8-1.5%. Worth ~3-5% bill if the closure tolerates it; needs a sub-tier implementation (no lazy trick).
- rank_dial.py (whole-tier rank, native): 352 -> util -1.5% / MSE +4.7% (r35), +3.8% (r100); 320 -> util -6% / +9.5%, +14.8%;
  288 -> util -7.6% / +21.6%, +24.8%. The tier rank is co-adapted at 384 (a 0.3% static plane error costs +4% raw: the closure
  amplifies plane errors ~10x more than 504aldo's eps law). => nested sub-frame at 256 (0.2-0.5% plane error) would be a wash without refit.
- NEW LINE (fable_refit_v1 chain_rank256.log): rank 256 for the whole tier (native util ~-10%) + JOINT REFIT (lost content ~1% of the
  plane, unlike the 60% of the drop line). Calibrations rank 288/256 on VALID8, then refit_rank256_lr2e4 (3 epochs), then the
  refit-only control refit_drop0_lr1e4 (2 epochs). refit_drop9 killed after epoch 0 (VALID 4.7e-8, 2.57x control: cannot pay).

## Switch to Astra's threads (Matt, ~16:10 machine time): "[quote removed]"
- Killed my rank-256 refit chain (epoch 0 unfinished; rank256 native util -11.5% / raw +38-43% before refit; train mean 3.68e-8 at update 10).
- Launched Astra's learnable_joint_basis_native_bridge_v1/bridge.py (his sealed pipeline: build -> null controls -> trained controls ->
  VALID40 parent/trained pairs -> seal -> score_native40; ~87 calls). Expect negative (pass-2/pass-1 0.995 vs needed ~0.94).
- His next levers: late-head residuals (455 values, joint consumer adaptation) and the 97-parameter joint-remainder carrier (needs
  diagnostics + differentiable impl + fit). My refit.py (all groups jointly) is the general form of his late-head idea: run the
  refit-only joint retune (drop 0, lr 1e-4) after his evaluation retires.
- Astra's engineering frontier (all_positive_leaf_catalog_v1, lazy_mixed_fmm_reprice_v3): fast-multiplication catalogs over the 295
  outermost matmul calls/net give OPTIMISTIC bill savings of 13.5-14.7% (executable Smirnov + bank Winograd) and 15.2-16.2%
  (extended 45-program), i.e. at/above the current 3e-9 uniform cut (14.47%), BUT actual Smirnov schedules were 8x slower in
  RESIDUAL time (leaf GEMMs 0.29 s vs a 0.15 s allowance); the bank's platform residual is already 0.30-0.385 s against the 0.4 s
  cap (zero-prediction fallback beyond it). Engineering is residual-capped, not FLOP-capped. Request diet (body ~23k requests/net,
  ~6 us residual + ~1.2 ms platform wall each) is the only way to buy room for deeper schedules.
- Chained: after Astra's bridge (PID 37968) retires, refit_drop0_lr1e4 (joint retune of P,A,cstar,edges; 2 epochs) = the general
  form of his late-head "joint adaptation" lever.
- Astra's learned-basis native VALID40 (his sealed pipeline, 87 calls, 47 min): raw ratio 0.99859, adjusted 1.06464, 0/40 wins ->
  CLOSED NEGATIVE (recorded in his index/README). Joint retune refit_drop0_lr1e4 running (chained). Sanity test of paired_eval.py
  (cand_drop16 vs bank on rank 37) and a per-site request census (callsite_census.py, r35) launched alongside.
- LINE A refit_drop0_lr1e4_w20 (joint retune of the unchanged card, warmup 20): VALID40 epoch 0 = 1.9815e-8 vs 1.9415e-8 before
  (+2.1%, 8/40 wins); train flat at ~1.92e-8 -> the deployed closure is at its optimum; no free co-adaptation slack. KILLED.
- PARALLEL LINES now: B = refit_rank256_lr1e4_w20 (rank-256 tier + joint refit, 3 epochs); G = refit_cumhead.py cumhead_joint
  (per-unit k3/k4 multiplicative corrections 0.3*tanh(h.w+b) from the causal GNN state at layers>=8 and the final map, 130 new
  values, trained jointly); H = cumhead_v_joint (G + per-unit variance correction 0.003*tanh and per-unit K22 scale sqrt((1+c4_j)(1+c4_k))).
- build_cumhead_variant.py: native cumulant-head hook (pair map at l>=8 + final scalar map; Cpre/K21/k4/k22 modified for the pair
  call only, like the torch wrapper). Zero-head parity on r35: MSE 1.648017257e-8 exact, util .18831 (+0.016%).
- Chains armed: G -> cand_cumhead_G.py + paired_eval VALID40 (paired_cumhead_G.json); H -> cand_cumhead_H.py; B (rank256, after
  epoch 2) -> cand_rank256_B.py + paired_eval. Bank 14-net panel baseline running (panel_bank331824).
- Rebuilt builder (K21 scaling + from-layer options) zero-head parity on r35: exact (1.648017257e-8). Three concurrent refits run at
  ~100 s/update each (contention) -> ~65 min/epoch. autoresearch/fire_candidate.sh = submit_cand.sh + ledger entry + receipt.
- Early readout (update 2-4): joint lines jolt +16% even with warmup (closure hypersensitive to 1e-5 steps); heads barely moved
  (norms 1e-4). Launched G0 = heads-only (k3/k4/v/K22 heads, closure frozen), lr 1e-3, warmup 5, 2 epochs, 4 threads, with its own
  build/paired/gate chain. Four refits now share 16 cores (~110-140 s per update each).
- Bank 331824 panel baseline (panel_candidate.sh, independent 4M truths): panel8 raw 2.042e-8 / util .1889 / adj 3.863e-9;
  virgin6 raw 1.666e-8 / util .1884 / adj 3.145e-9 (per-net raw in fable_refit_v1/panel_bank331824.log). gate_and_fire.sh requires
  both panel8 and virgin6 adjusted < 0.98x these.
- head_info_probe.py (r35 fit / r100 transfer, 8M truths): the card's per-unit k3/k4 relative errors (rms 0.14-0.24) and v error (0.07-
  0.16%, = truth noise) are NOT predictable from the GNN state h: ridge R2 0.05-0.09 in-sample, negative transfer; tanh-MLP 0.1-0.3
  in-sample, negative transfer. => per-unit cumulant heads have no usable information; lines G/H KILLED (joint variants also jolt the
  closure); G0 (heads-only) kept as the cheap control. Launched B2 = rank-320 tier + joint refit (native util -6%, raw +10-15% before).
- Astra's joint-remainder carrier (97 params, _jr_* functions, _JR_LITERAL theta/phi/head/head_b) is the PARENT of the learned-basis
  line: the basis fit trained the 97 + 3072 values and closed negative natively (raw -0.14%) -> the carrier thread is finished.
- Data: the HF cache holds only phase-1 shards (32x256x256) + minis; the phase-2 full dataset (1000 nets, frozen 700/100/100/100
  split) is governed by data_gate/ (custodian-projected all_layer_means targets); truthpost targets exist for ranks 35-234 only;
  the STORE has weights for 688 training-role nets. A "more training nets" refit would need the custodian projection (authority.json).
- Live: B rank256 refit (update 8, train 2.53e-8 vs 2.59e-8 pre), B2 rank320 (VALID40 pre 2.08e-8 = +7.3%), G0 heads-only (norms
  growing 1.6e-3); B3 rank352 queued after G0. Automation: run_chain.sh (lock) -> paired VALID40 -> gate_and_fire.sh.
- edgeworth2_test.py (torch card, no refit): adding the k3^2 He6/72 second-order marginal terms (m, r2, r3, r4) -> r35 1.6489e-8 ->
  1.6908e-8 (+2.5%) at scale 1, 1.6606e-8 (+0.7%) at 0.5; r100 +3.1% / +0.7%. The closure compensates for the missing term; no
  free win (would need a refit; expected <= 1-2%).
- series_order_test.py: pair-series truncation (Gaussian order 6->8/10, correction terms 3->4/5) changes the final MSE by <= 0.08%
  (r35 1.6489e-8 -> 1.6476e-8 at 10:5). The series is converged; dead dial.
- learned_off_test.py (torch): GNN mean hook zeroed -> r35 1.981e-8 (+20%), r100 2.095e-8 (+24%); covariance edges zeroed -> +4.7%/+4.0%;
  both -> 2.038e-8 / 2.157e-8 (+24%/+28%). The learned closure carries ~19-22% of the accuracy; the physics carries the rest.
  A from-scratch joint refit after a physics change would have to relearn ~20% (possible in a few epochs) — the only way to test
  physically-right changes (Edgeworth-2, truth-like planes) that the co-adapted closure punishes.
- LINE S launched: refit_scratch_lr1e3_rest1e5 = GNN head/head_b and edge output/bias ZEROED, then joint training (zeroed groups lr
  1e-3, all other groups 1e-5, warmup 10, 3 epochs, 4 threads). Question: can the refit relearn the ~20% the learned parts carry?
  If yes -> "physics change + from-scratch refit" is viable; if not -> the refit paradigm is the limit.

## Gain-mixture (b-vector) mechanism (2026-09-22 evening, Opus 5.5 session)
- k4 per unit ~= a v^2 + 0.7 k3^2/v (8M truth R2 0.71-0.89 vs 0.58-0.76 isotropic at L12-15). As a card supply (torch, no refit):
  +/-1-3% by net -> parked (needs refit).
- K21 plane error on 331824 (r35 L12-14 vs 8M truth): 63-68% of error energy rank-1; left ~ sd (cos .92-.95), right ~ mu (cos .955-.965)
  (mu ~ top eigvec of Cpre, cos .95-.97). One scalar kappa_l (4.8-5.6e-3) x Sym(C,mu)_(2,1) = v_a mu_c + 2C_ac mu_a explains 56-64% of
  the error energy (rel err .61-.64 -> .39-.40). The card's own b-term is aligned (cos .91) but ~2.6x too small; b_along_mu + kappa =
  0.84-0.90 x radial/6.
- MECHANISM: for a common gain mixture (ReLU homogeneity: h = G z), K3 contains (radial/6) Sym(C x mean). The card's b recursion
  (b_post = g*bpre + radial*sigma*phi/6) only adds the kink part of each radial increment, and at depth sigma*phi is <2% of m, so b
  stalls at 0.32-0.41 x radial/6 (replayed from dumps, recursion exact to 3e-8). The missing part is absorbed by the star (plane-only
  carrier, approximate transport) -> the rank-1 plane error.
- Fix b_post += gamma*(radial_new - radial)/6*m (model_bank_bfix.py): torch no-refit final MSE on 35/100/37/39: gamma .5 +8%, .85 +34%,
  1.0 +52% (co-adaptation, as with truth planes). Checking plane accuracy under the fix (plane_bias_map_bfix.py) to decide on a refit.
- Killed refit_rank256 (update 28: train 2.60e-8 = no recovery).
- PARTITION b-fix REFUTED: physics-only card (GNN hook + edges zeroed) also degrades (gamma .5 +15%, 1.0 +60%); rank-35 plane residual
  unchanged at gamma .85 (kappa L6 2.26e-3 -> 2.16e-3): the star absorbs what is added to b at post-activation (re-partition only).
- ADDITIVE b (extra term carried in b, partition unchanged): the plane becomes right (gamma .5: rel err L8/9/12/13 .60/.63/.62/.60 ->
  .47/.47/.44/.44, kappa residual ~0) but means explode 10-20x (L8 6.8e-9 -> 1.6e-7, learned on and off): the full Sym(C,m) tensor also
  adds 3 v kappa mu to each unit's pre k3 (Delta m ~ kappa*sigma*alpha^2*phi/2 ~ 4e-4), while truth's per-unit k3 has NO mu-structured
  error (diag kappa R2 ~0). => the missing structure is Sym(C,mu) OFF the diagonal only (a K21 (2,1)-slice term, not a full b-family tensor).
- NEXT (running): K21 off-diagonal-only correction kappa_l*(v_a mu_c + 2C_ac mu_a), diag zeroed, kappa_l = max(0, gamma*radial/6 - bpre.mu/mu.mu)
  (derived; matches fitted kappa at L8-14 within ~5% for gamma .87), variants: pair map only / all consumers (bfix_mode_test.py).
- resp_floor_test.py (response-row floor, reads = 26% of bill): floor .02 -> util -5.6%/-2.4% (r35/r100), MSE +9.2%/+4.1% (adj +3.1%/+1.6%);
  .03 -> +20%/+26% adj; .05 -> +86%/+107%. Another cliff; closed.
- star_k31_truthcheck.py (one step from the MC-true state, r35): the exact Gaussian-born STAR term of K31 has joint-fit coefficient
  0.86-1.08 (physically right) but explains only +1-2 pts of R2 of the true K31 (iso .43-.55 -> both .44-.57; star rms 13-16% of truth).
  The -12% truth-K31 gain is NOT the one-step star term.
- deadrow_test.py: zeroing DEAD rows (gate<.01) of the cube bank after the push (would save ~5-6% of bill if skipped in the old
  reconstruct + young push): old-tier rows only +4.4%/+7.1% raw (r35/r100), all columns +11.5%/+20.7%. Dead units' k3 matters
  (their small means feed mu_next at the 1e-4 level). Closed.
- star_k31_test.py (exact one-step star term of K31, closed loop, learned ON, no refit): L2-14 add lam1 -1.45% (4/4), L11-14 -0.75% (4/4);
  cost ~0.36% bill/layer (two n x n_sel x n reads) -> net +0.7..+4% adjusted. Right physics, too small.
- ZERO-COST PHYSICS (physicist's per-row K31 factor r_a = .85 + .5 mu_a k3_a/v_a^2 on the isotropic T31; my k3-derived k4, beta .5):
  learned ON: k31 -1.52% (.9872/.9834/.9991/.9695 on 35/100/37/39), k4 +0.66%, both -0.34%; learned OFF: k31 -3.41%, k4 -3.94%, both
  -7.2% (4/4, -5..-10%). The learned closure neutralises ~5 pts of physics gain. RUNNING: heads-only retrain arms (control vs physics,
  TRAIN32/VALID16, 2 epochs) = refit_physics.py; porter's paired VALID40 of the K31 row factor (eval_t31.sh, cached bank rows).
- engineer RESULT: kernel request diet -22% warm requests (bit-exact) but bill flat; deeper Strassen <= 2-2.35% (divisibility/wall);
  view-binding reshapes 4.0%/1.8% of the bill (first/second net). Follow-ups assigned: free view binding; 64/128 rounding for depth.

## 2026-09-22 ~20:10 machine time — BOARD REALITY + frame placement (fable_frame_v1)
- LIVE BOARD (read 02:35 UTC 09-23): 1 J2W 2.1e-9 (raw 1.64e-8 @ .1307), 2 suliman_tadros 2.2e-9 (1.63e-8 @ .1376), 3 marius_binner 2.4e-9
  (1.68e-8 @ .1459), 4 a_s6 2.5e-9 (2.00e-8 @ .1258), 5 Puffi/reds 2.8e-9, 7 oqaris/ben3 3.1e-9, 9 Camaro 3.3e-9, 10 US 3.5e-9 (1.91e-8 @ .1838).
  The leaders beat us on BOTH axes (raw -14%, util -29%). 504aldo (public post): leaders' bill = a K3 card minus its old tier.
- frame_test.py (torch, learned ON, no refit): the shared old tier's frame chosen AFTER the push (range finder on the pushed Gram in
  pre-activation space) vs the bank (post space, before the push); and exact top-r eigenvectors instead of the one-pass sketch.
  r35 / r100 final-MSE ratio vs bank rank 384:
    sketch post (bank): 384 1.000/1.000, 320 1.094/1.145, 256 1.384/1.428
    sketch pre:         384 1.012/0.993, 320 1.086/1.073, 256 1.288/1.155
    exact eig post:     384 1.015/0.968, 320 1.043/1.035
    exact eig pre:      384 1.014/0.978, 320 1.064/1.013, 256 1.178/1.242
  => better frames halve the rank cliff but do not remove it: the best rank-320 frame is +3.9% raw (2-net mean) for a native -6% util;
  an exact/power-iterated frame costs ~1-1.5 units/join-layer (~+5-8% bill), so the frame line is a WASH. CLOSED. The old tier needs
  ~384 dims per leg; its content is generic in that frame (~r^3/6 dof), so the read floor is ~n_resp*Nold*r.
- Heads-only retrain arms (fable_refit_v1/refit_physics.py, TRAIN32/VALID16): epoch 0 valid: control 1.8853->1.8849e-8, physics
  (K31 row + k4 zero-sum) 1.8837->1.8746e-8 = -0.55% vs control. Re-adaptation releases ~nothing beyond the zero-cost K31 row.
- Native K31 row factor (porter, c0 .85 c1 .5 L1-14): VALID40 adj 0.98606, 38/40 wins; stacks bit-exact on engineer2's e2 build.

## 2026-09-22 ~21:00 machine time — exact-state oracle map (fable_frame_v1) + coincident-index K4 (fable_k4diag_v1)
- MEAN reset to the 1e9 truth after layer l (card unchanged after, learned ON): r35 .992(L1) .982(L4) .937(L6) .881(L8) .747(L10) .543(L12)
  .415(L13) .232(L14); r100 identical shape (.781 L10, .544 L12, .414 L13, .232 L14). => the final error is BORN over L9-15: ~13% L9-10,
  21% L11-12, 12.5% L13, 18.5% L14, 23% L15 (r35). Final error lives in ACTIVE output units (true mean > .2: 82% of MSE).
- COVARIANCE reset (32M truth at L4/7/10/13): all C at 4 layers .765/.785 (r35/r100); ONLY the part orthogonal to the card mean
  (P dC P, P = I - m m^T/|m|^2) carries it: Cperp@4+7+10+13 .740 (r35), Cperp@13 .888/.844, Cpar (the rest) 1.015-1.019.
  => kink units (mu_a = w_a.m ~ 0) see only P dC P. The card's K21 error (63-68% rank-1 along mu) lives in the mean direction =>
  explains why truth K21 in the covariance channel does not help.
- P dC P off-diagonal rel error vs ||P C P||: .44% (L4), .55% (L7), .69% (L10), .86% (L13); 32M noise .31/.24/.20/.18% => real at L7+;
  BROAD (top-16 SVD 25-44%, top-64 60-79%); no scalar multiple of C, C^2, C o C, corr, C^3 explains > 1%. No cheap global fix.
- PER-UNIT exact pre cumulants (indep 32M; card mean kept), r35: v@15 .946, k3@15 .963, k4@15 .972, vk3k4@15 .859; vk3k4@14 .910;
  vk3k4@14-15 .785; v@10-15 .795 (!), k3@10-15 .917, k4@10-15 .919, vk3k4@10-15 .651, vk3k4@2-15 .556. r100 v@15 .887.
- COINCIDENT-INDEX fourth-cumulant push (sum_b W_ba^3 W_bc kappa4(y_b), rectified-Gaussian kappa4 certified vs MC): 2-6% of the truth
  K31 scale, adds <= 0.7 pt R2 -> DEAD. Isotropic forms explain K22 94-98%, per-unit k4 94-98%, but K31 only 40-50% (r35 L4-13).

## 2026-09-22 ~21:40 — JOIN SCHEDULE of the shared old tier (fable_frame_v1/join_schedule_test.py torch; fable_join_v1/build_join_schedule.py native)
- The bank joins one cohort per layer (L5..14) into the rank-384 frame; every join re-truncates ALL old content to 384 dims. Joining
  LESS often keeps cohorts exact longer and truncates fewer times -> MORE accurate. Torch (learned ON, no refit), final-MSE ratio:
  | schedule | 4 std nets (35/100/37/39) | 12 TRAIN nets | native bill r35 cold / r100 warm |
  | even: 2 cohorts @ L6,8,10,12,14 | +2.0/-2.7/-1.9/-1.0 (mean -0.9%) | -2.0% (12/12) | -0.78% / -0.44% |
  | every3: 3 @ L7,10,13            | +0.4/-4.2/-2.1/-1.5 (mean -1.9%) | -2.2% (10/12) | -0.76% / 0.0% |
  | every3b: 3 @ L8,11,14           | +2.1/-3.7/-2.1/-2.0 (mean -1.4%) | -2.9% (10/12) | +2.1% / +3.5% (7 young cohorts at L7/10/13: too dear) |
  | every3c: 2,3,3 @ L6,9,12        | - | -1.8% (10/12) | - |
  | late2 (singles L5-8, pairs L10/12/14) | mean -0.5% | - | - |
  | every3 at rank 352 / 320        | +1.5/-1.5/-0.8 / +1.4/+5.1 | - | (rank 352 ~ -1.5% util) -> lower rank eats the gain |
  The native build must re-time the bank's lazy factor rotation (_slr_*: lazy at 8/10/12/14, consolidate 9/11/13 hard-wired);
  the builder does it; 'every' reproduces the bank bit-exactly. The exact old tier (no frame; physicist) is +1.7% r35 / -4.0% r100:
  every3 ~ reaches it. Net adjusted of even / every3 ~ -2.4% each; torch VALID40 screen running (join_schedule_test_valid40.log).

## 2026-09-22 ~23:00 — WHERE THE BIG LEVER IS, AND WHY NOTHING CHEAP REACHES IT (analyst, covhunter, physicist)
- Exact carried covariance at ALL layers: -29.6% (r35) / -35.3% (r100); at half strength -31/-37% (overshoot law); per single layer
  L8-12 worth -10..-21% each (covhunter). Its value flows 100% through the pair map (post covariance -> later per-unit variances).
- The covariance error is injected by the pair map's plane inputs: truth K21+K22+K31 in the covariance output (mean-orthogonal)
  = .762/.746 (the whole lever); all pair inputs at eps .5 = -26/-27%. One-step injector ranking: kink-row K21 read, then K31.
- The error that matters is one-sided in the top 30-100 eigen-directions of the card's own P C P (93-96% of the gain).
- NOT predictable from the card's own state: full plane correction 5-7% cross-net, top-k block 5-9%, carried error in the subspace ~1%;
  applied predictions hurt. Physical top-k supplies (P1, gain-skew, trees) <= -0.18% closed loop; built from TRUTH K21 they score
  WORSE (their alignment is the card's own K21 error).
- STRUCTURAL CONCLUSION (physicist): a k-direction read of what the card carries adds nothing (the card already forms the full K21
  plane from those carriers). The missing information is the NON-ISOTROPIC FOURTH CUMULANT (isotropic K31 explains only 40-50% of
  truth; trees explain 12-13% of the rest even with transport) plus its feed into the next layer's third cumulant. A carrier of
  that content (e.g. K3 x K3 second-order channel) needs ~n^2 x Ncol per direction: not cheap. This is the frontier.
