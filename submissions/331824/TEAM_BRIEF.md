# TEAM BRIEF — ARC White-Box Estimation, Phase 2 (width 1024, depth 16) — lead: Fable/Claude (main session)

## Goal and score
- Score per network = final-layer MSE x max(0.1, metered_flops / 2**41); board score = mean over networks. Lower is better.
- Official bank: submission #331824, adjusted 3.507e-9 (raw MSE 1.9006e-8, util ~0.1845). Milestones: 3e-9, 2e-9, ~1e-9.
- Board top (2026-09-22): 2.1e-9 (raw 1.64e-8 at util 0.131), 2.4e-9, 2.4e-9, 2.5e-9 (raw 2.00e-8 at util 0.126). We are rank 10.
- 3e-9 needs -14.5% adjusted: e.g. -14.5% util at the same predictions, or -14.5% raw at the same util, or a mix.

## Hard rules (disqualification-grade for the submitted estimator)
- The estimator runs ONLY flopscope ops (no numpy at eval), <= 2**41 FLOPs, <= 120 s platform wall per MLP, residual (non-flopscope
  Python) <= 0.4 s per MLP (else zero-prediction fallback), <= 8 GB. Platform wall ~= 1.1-1.4 ms x flopscope requests + ~2 x local compute.
  The bank's worst platform wall was 110 s and worst residual 0.385 s: there is very little headroom.
- NEVER edit frozen work: codex_resume_20260919_v1/*, astra_phase2_reexamination_v1/*, fable_wall_repair_v1/* originals. Create new
  files/dirs (fable_<lane>_v1/). Never delete; version instead. ONLY the lead submits to the board. Do not run git commit/push.

## Machine (16 cores, 48 GB, shared with the lead's jobs)
- Always: nice -n 10; set OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS/VECLIB_MAXIMUM_THREADS to your lane budget; torch.set_num_threads(budget).
- Check `memory_pressure | tail -1` before launching; do not launch if free < 35%. Kill anything above 9 GB RSS. No huge files (> 2 GB) without need.
- Lane thread budgets: engineer 2, porter 2, analyst 3. The lead uses ~8.

## Paths (all under pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/, call it P2)
- Bank candidate (read-only): P2/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py (live loop = LAST
  definition of _predict_binary; Estimator.predict -> _predict_learned_compact -> _predict_binary).
- Weights: P2/learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3 (manifest.json
  roles.observer_fit.row_manifests[rank].path -> weights.npy, (16,1024,1024) f32). Ranks 35..234 are TRAIN160/VALID40
  (P2/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json 'train'/'valid').
- Truths: final/all-layer means P2/public_g22_operator_features_v1/truthpost/truthpost_rank{r:04d}.npz ['mc_post'] (16,1024), ranks 35-234.
  Per-unit 8M truths P2/public_g19_attribution_v1/mps_truth_rank{r:04d}_P8M.npz (mc_pre_m/v/k3/k4, mc_post, mc_post_var, mc_se) for
  35,100,150,200,250,300,400,500 (P32M for 35,100). (2,1)-plane truths: P2/public_g19_attribution_v1/hub_planes_rank{35,100,150,200,250}_honest.npz
  ['pre_l'] = K21[a,c] = kappa3(h_a,h_a,h_c) at layers l=1..14 (same orientation as the card's K21); mps_planes_rank0035_new.npz (K21/K22/K31_12..15).
- Torch research card (differentiable port of the bank; L15 fidelity 4.2e-6 RMS vs native): P2/fable_refit_v1/model_bank_research.py,
  gain-mixture variant model_bank_bfix.py (module globals BFIX_GAMMA, BFIX_MEAN). Python: /Users/matt/anaconda3/bin/python (torch 2.7).
  Examples: fable_refit_v1/{refit.py, bfix_test.py, plane_bias_map.py, learned_off_test.py, k4_from_k3_test.py}. Forward ~10-17 s/net.
- Native (flopscope) tools, Python /Users/matt/arc-white-box-estimation/.venv/bin/python: fable_refit_v1/score_variant.py <cand> <ranks>
  (util + final MSE), paired_eval.py <cand> <parent> <ranks|valid40> <jobs> <out.json> (own bills), export_params.py (refit .pt ->
  candidate literals, bit-exact round trip verified), build_cumhead_variant.py (example of patching the live loop), gate_and_fire.sh.
  Bill census: P2/fable_wall_repair_v1/bill_census.py <cand> <rank> [callers]. Request census at the production boundary:
  P2/fable_wall_repair_v1/run_multinet_rss.sh (numpy-free client P2/data_gate/.client-venv + OpenBLAS server) and callsite_census.py.
- Notes of today's measurements: P2/fable_oldtier_economics_v1/NOTES.md (read it first). Astra's history: P2/codex_resume_20260919_v1/REEXAMINATION_INDEX_20260919.md.

## Current science (2026-09-22)
- Every local bill dial on the bank is a cliff (old tier, births, star, compact pushes, rank); joint retune alone: no gain.
- NEW MECHANISM (lead): the card's (2,1)-plane error is 63-68% rank-1 ~ Sym(C, mu) (left ~ sd, right ~ mu). The card carries a
  Sym(C x b) family in b (exact transport); for a common gain mixture K3 contains (radial/6) Sym(C x mean), but the card's recursion
  b_post = g*bpre + radial*sigma*phi/6 adds only the kink part of each layer's radial increment, so b stalls at ~1/3 of radial/6 and
  the star (approximate transport) carries the rest. Fix: b_post += gamma*(radial_new - radial)/6 * m. Without refit the final MSE gets
  worse (gamma .5 +8%, .85 +34%, 1.0 +52% on 4 nets) — the learned closure is co-adapted to the old error. Plane-accuracy check running.

## Reporting
- Each lane writes P2/fable_<lane>_v1/RESULT.md (what ran, numbers, verdict) and appends short dated lines to P2/fable_refit_v1/TEAM_LOG.md.

## MPS RULE (Matt, 2026-09-22 ~21:15): give MPS a rest. After covhunter's current r39 all-layer MC finishes, NO new MPS jobs by anyone until Matt says otherwise. CPU only.

## OVERSHOOT LAW (covhunter, 2026-09-22 ~21:45) — grade every physics fix at PARTIAL strength
Truth supplies fed to the pair map's covariance output as card + eps*(truth - card), closed loop, learned ON, no refit:
kink-row K21 L6-14 eps .25/.5/1 = -7.7/-9.6/+4.1% (r35), -7.3/-8.7/+6.2% (r100); K31 L2-14 eps .5/1 = -13/-13% (r35), -14/-15.5% (r100);
all pair inputs L6-14 eps .5/1 = -26/-24% (r35), -27/-21% (r100). The card's plane-driven error is anti-correlated with the rest
(learned parts compensate), so full fixes overshoot (quadratic model: first-order adjoint price -42%, curvature ~.45 -> eps* ~ .45).
RULE: test any change at eps in {0.25, 0.5, 1} (blend of new and old), and report the adjoint first-order price when you can.
