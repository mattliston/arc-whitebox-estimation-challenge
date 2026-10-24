# ASTRA TAKEOVER — ARC White-Box Estimation Challenge, Phase 2

Written 2026-10-10 ~23:20 UTC by the outgoing lead (Claude Opus). Read this file first, then the newest ~150 lines of
`O/TEAM_LOG.md`, then each lane's `O/<lane>/TAKEOVER_ASTRA.md`.

`O` = `/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`

## 1. Goal

Matt: "it's extremely important that we win this competition". Aim for FIRST place in Phase 2. The deadline is
**2026-10-17 23:59 UTC**. Matt thinks a further ~-9% is reachable at our recent pace (-5% S42 -> S45 in 13 hours).

## 2. Board (2026-10-10 ~22:00 UTC)

| place | team | adjusted score | final-layer MSE | budget share |
|---|---|---|---|---|
| 1 | J2W | 1.4425e-9 | 1.3545e-8 | 0.1065 |
| 2 | marius_binner | 1.5099e-9 | 1.159e-8 | 0.1303 |
| 3 | Luna | 1.5612e-9 (#335221) | 1.3458e-8 | 0.1160 |
| 4 | **us (S46)** | **1.581e-9** | 1.354e-8 | 0.1168 |
| 5 | oliver_chen | 1.6081e-9 | 1.2304e-8 | 0.1306 |

(Analyst scan 21:59Z, `O/analyst/a5_leaders_20261010T2159Z.md`; our S46 graded after that scan.)

- **Gaps:** to 3rd we need -1.25%, to 2nd -4.5%, to 1st -8.8%.
- **Where the gap to J2W sits:** our final-layer error equals J2W's; the difference is bill.
- **Loss risk (analyst model):** J2W loses a network in about 34% of fresh 100-network runs; its walls are 107-116 s. Our risk is 1-2%.
- **First place (`O/analyst/a16_pair_mc.py`):** P(1st) is small until we pass marius. After that it rests on J2W failing, until about -9% beats J2W outright.

## 3. Our submissions (exact archives that were uploaded)

The full record is `/Users/matt/arc-white-box-estimation/autoresearch/SUBMISSION_LEDGER.md`: one FIRE line and one GRADED line per submission, with archive and source sha.

| file | id | score | archive |
|---|---|---|---|
| S46 safe | #335259 | 1.581e-9 | `O/release/runs/a_s46_63f89e8d_safe_pz/submission.tar.gz` |
| S46 twin (engineer) | #335261 | 1.581e-9 | `O/release/runs/a_s46e_f89b8951_primary/submission.tar.gz` |
| S45 safe / twin | #335213 / #335215 | 1.586 / 1.584e-9 | `a_s45_b36b7154_safe_pz`, `a_s45p_primary` |
| S44 safe / twin | #335147 / #335148 | 1.629 / 1.627e-9 | `a_s44_67dfcf71_safe_pz`, `a_s44p_primary` |

The receipts and graded pages are in `O/fire/<tag>/` and `O/board_evals/eval_<id>.json`. Six submissions were used on 2026-10-10 UTC.

## 4. Rules that bite

- **Score:** final-layer MSE x max(0.1, billed flops / 2^41), mean over the networks.
- **Code:**
  - All compute goes through flopscope; numpy compute is prohibited.
  - No underpriced-op tricks: the organisers may disqualify for them.
  - Read every NEW einsum signature before it ships. Distinct operands only, never a repeated operand.
- **Limits per network:** 120 s, 8 GB, 2 vCPUs, and a residual of at most 0.4 s.
- **Setup cap:** a one-time setup over 5 s FAILS THE WHOLE SUBMISSION. Ours project to 4.30-4.31 s against our judge line of 4.5 s.
- **Submissions:** 10 per UTC day.
- **Designation:**
  - We may designate up to TWO files, and only the BETTER one is ranked.
  - So designate a pair: the safe file plus its twin, same physics.
  - If we do nothing, the default is our two best public scores.
  - The selection interface opens "near the end"; the analyst's hourly watch reports it.
  - Organisers announced a flopscope re-release with re-evaluation before the selection. Re-measure our files if it lands.
- **Secrets:** the API key is in `/Users/matt/arc-white-box-estimation/key.txt`. Never print, copy or log it; only `O/fire/fire_once.py` reads it.
- **Machine:**
  - 16 cores, 48 GB, one GPU. Never cause an OOM.
  - Keep the load sane while the release lane is timing stage C, which measures wall time.
  - Never delete or overwrite anything: version and archive instead.

## 5. How a gain becomes a submission (the pipeline that produced S44-S46)

1. **TRAINER** (torch card, `O/trainer`) runs refit rounds:
   - Rounds are `jmix*.sh` drivers. The arms are learned arrays, head-input rows and physics dials.
   - Step sizes are fitted on one fresh 16-network set and accepted on a second, independent 16.
   - Each round is judged ONCE on the 48 clean judging ranks.
   - The output is a hand-over in `O/trainer/candidates/<name>/`: candidate.py, physics_dials.json, head table.
2. **PHYSICIST** judges on UNTOUCHED generated networks (`O/physicist/t79_fresh_judge.py`).
   - This is the primary evidence; the 48 ranks run optimistic.
   - Reference points: board -2.6% vs untouched -1.3% (S45); board -0.3% vs untouched -0.66% (S46).
3. **ENGINEER** builds natively: `bash O/engineer/stack/s43_build.sh <hand-over> <head table> s46s` (safe base) or `s46p`.
   - Then the per-layer parity check on networks 651 / 378.
   - Then ten fresh networks against the previous native file.
4. **RELEASE** qualifies each file:
   - SAFE = sha + Rl3_bind128 + Rlvfs_limit_off + Rprices_zero; also the TWIN.
   - Stages: stage A (einsum list), stage C, 1.3x slow-setup session, `designation_judge.py` lines.
   - It writes a "FIRE LINE ..." or "DO NOT FIRE" line in TEAM_LOG.
   - Stage A cannot see errors at layer 10+; stage C can.
5. **LEAD** checks and fires:
   - The archive's estimator.py sha must equal the candidate sha.
   - No einsum signature may be new against the last fired file.
   - Fire with `cd /Users/matt/arc-white-box-estimation && .venv/bin/python -B O/fire/fire_once.py <tag> <archive> "<text>"`.
   - Then `nohup bash O/fire/poll_eval.sh <tag> <id> 150 &`.
   - Ledger FIRE/GRADED lines, TEAM_LOG line, copy `eval_<id>.json` to `O/board_evals/`.

## 6. What has worked (keep doing it)

Every gain this week was NEW INFORMATION for the learned parts, or a re-fit of something that had gone STALE because
the rest moved:

| file | change | gain |
|---|---|---|
| S42 | law-diagonal head input | -2.6% |
| S44 | joint refit + column 17 at layer 15 + padded dead cut | -2.3% |
| S45 | physics dials re-solved after the arrays moved (X1 birth-leg dial 0.557 -> 0.708, c0 / k31) | -2.6% |
| S46 | iso / K4L per layer at layers 10-14 | -0.3% on the board, -0.66% untouched |

After each big move, re-screen everything with zero-step gradient tables: dials, edge arrays, head rows.

## 7. In flight at handover (all running in the background)

- **jmL (trainer):** the learned covariance EDGE arrays were found stale (rho_scale mean t^2 26.8, bias 10.7, left 7.1). jmL gives them their own step, separate from the head arrays.
  - Log: `O/trainer/cert/refit/jmix_L.log`. Step-1 verdict about 00:00 UTC.
  - Then judge on the 48 and the physicist's untouched seeds, export, engineer build, release qualify, fire as a pair.
- **Layer-15 unmask (physicist):** at layer 15 the head masks 10 of its 30 inputs. Unmasking them with fixed weights LOSES (+0.3..+4.4%).
  - A 55-network dump is running (`O/physicist/t80_l15dump.log`) for an offline refit of the layer-15 head on those inputs (`t83_l15_unmask.py`, `t84_l15_ceiling.py`), judged on 32 fresh seeds.
  - The native switch `U15a` is ready: +0.28% bill for group (a), features 13-15 + 26.
- **NEXT LEVER, queued for the trainer after jmL: the cut-simulated refit.**
  - The native file is 0.6-1.2% worse than the torch card at every step. 91% of that comes from three native cuts that are absent from the torch training card: old-tier tails, young dead tail, pair-mean truncation.
  - Each cut pays its own bill, so keep them, but refit the learned parts WITH them: `O/engineer/a22_cutsim/torch_cutsim_v2.py`.
  - Certify first that the cut-simulated torch matches the native predictions.
- **Small item:** the b-fix gamma constant, 0.1 -> 0.15-0.2, as a dial, about -0.2%.
- **Analyst background jobs:** fresh generated networks with 64M-sample truths on the GPU (about 9 an hour, seeds 7100097+); the hourly watch (selection interface, flopscope release, forum); board scans every 4 h.

## 8. Fresh-seed assignment (do not mix)

- Seeds 7100000-7100103 are used.
- 7100104-7100135: the physicist's S46 untouched judging (used).
- The trainer's sizing / acceptance sets come from 7100136 onward, if his engine reads generated seeds.
- New untouched judging: the next unused block. Write the assignment in TEAM_LOG before using a block.

## 9. Closed on 2026-10-10 (measured; do not redo)

- **Layer 15:**
  - layer-15-only weight refits (overfit);
  - layer-15 inputs other than column 17;
  - unmasking the layer-15 features without a refit.
- **Head inputs and refit variants:**
  - per-unit variance-error predictors (V15): the variance error is carried off-diagonal covariance, with no cheap signature;
  - qold, fold (+1.34% bill), batch-7 columns (no held-out gain);
  - band biases (jmF: failed the 48);
  - the multi-layer loss (closed 09-24);
  - frozen arrays (centers / scales / rms / gain: at their optimum).
- **Covariance terms:**
  - low-rank covariance corrections;
  - the Cholesky sandwich (billed too high);
  - Rayleigh-Ritz at the L10 join (+0.42% bill).
- **Capacity and representation:**
  - capacity reallocation and more births: every capacity cut costs more than its bill; adding is flat;
  - old-tier rank +64: 1:1 with bill, and no co-adaptation gain on the 384 base;
  - cheaper young carriers (earlier joins +40%, eigen-projected reads +10..+131%);
  - the |lambda|-weighted sketch;
  - the plane oracles' ceiling (-18% at layers 10-14): real, but no cheap carrier exists.
- **Bill tricks:**
  - half precision (float16 bills like float32);
  - the read-column magnitude cut;
  - transform reuse;
  - deeper Winograd.
- **Native approximations:** the full audit flagged none.

## 10. Team

Five lanes: trainer, engineer, release, physicist, analyst. Each has `O/<lane>/RESULT.md` and
`O/<lane>/TAKEOVER_ASTRA.md`. Astra may run agent teammates in the same roles. Lanes log one line each in `O/TEAM_LOG.md`, and every stamp comes from `date -u`.

The memory of all prior work is in `/Users/matt/.claude/projects/-Users-matt-arc-white-box-estimation/memory/` (MEMORY.md is the index).

## 11. Plan to the close

1. Each day: one or two verified gains, fired as a pair (safe + twin) when the untouched judge holds.
   - Matt approved firing improvements of "a couple of percent".
   - In this final week I also fired verified gains of -0.6..-2.6% that improve our place or the designated pair.
2. Keep the best pair graded at all times. Designate it when the interface opens, or rely on the default of the two best public scores.
3. In the last 24 hours, stop changing physics; only re-qualify and fire what is verified.

## 12. Update 23:22 UTC

- **Layer-15 unmask refit: CLOSED / no-go** (physicist T40).
  - Group (a) with new layer-15 weights reads -0.013% held out (se 0.028) against +0.28% bill; all ten inputs read +0.14%.
  - A flexible residual model reaches about 0. Keep the engineer's U15a switch OFF.
- **Remaining queue:**
  1. jmL (stale edge arrays): its verdict, then judge and build.
  2. The cut-simulated refit (`a22_cutsim/torch_cutsim_v2.py`; certify it against native predictions first).
  3. The gamma dial.
  4. Re-screens after each move.
- **Trainer note:** `O/trainer/TAKEOVER_ASTRA.md` may still be missing. If it is, read `O/trainer/RESULT.md` and the newest trainer lines in TEAM_LOG. The drivers are `O/trainer/cert/refit/jmix*.sh` (logs `jmix_<X>.log`); the pool is `O/trainer/worker.py` under its supervisor. Workers need `TRAINER_PL=1 TRAINER_CARD=M`, and the supervisor has a 30-restarts-per-index limit (it caused an outage at 19:30Z).

## 13. Update 23:30 UTC (last note from the outgoing lead)

- **jmL is CLOSED, no step.** The edge arrays' strong zero-step gradient does not pay at a finite step: sharp curvature, the same pattern as column 17 earlier.
  - The solved step read +0.052% on the independent check (se 0.16); the edge arm alone was +0.45% (se 0.22).
- **The trainer lane is converged at jmIv = S46** on all screened knobs.
- **The next live lever is the cut-simulated refit** (section 7; certify `O/engineer/a22_cutsim/torch_cutsim_v2.py` against native predictions first).
- **Layer 15:** the physicist's flexible-model ceiling says no layer-15 information predicts the remaining final-layer error, so final-layer gains must come from a better layer-14 state.
- **Trainer notes:** `O/trainer/TAKEOVER_ASTRA.md` was written by Astra itself. The outgoing Claude trainer's own note is beside it: `O/trainer/TAKEOVER_NOTE_CLAUDE_TRAINER.md`, 46 lines with the exact commands, request fields, protocol and lever list. Also see the handover section at the end of `O/trainer/RESULT.md`.

## 12. Astra update 2026-10-11T00:01:27Z

Authoritative continuation state is the newest `O/TEAM_LOG.md` plus lane receipts. The original handover's in-flight descriptions above are historical.

- jmL is terminal: its proposed step failed the independent 16 (+0.052%, se 0.162); no accepted point, judgement or export. S46 remains the current graded pair.
- Layer-15 unmask refit/ceiling is closed without a held-out gain (physicist T40).
- Old `engineer/a22_cutsim/torch_cutsim_v2.py` is not sufficient for current S46. A versioned replacement is being numerically certified before any refit. Disabled651 exactly reproduces all16 original predictions; generated104 bridge exactly reproduces the physicist's baseline. Enabledv7 failed (final native RMS5.65e-5 vs1.75e-5 originally); push-only improves slightly, row-only reproduces the fault. Do not train on a simulator merely because it passes static review.
- Seed assignment v4 supersedes section8 and every older lane assignment: 7100136-151 sizing,7100152-167 independent acceptance,7100168-199 untouched final judgement. Trainer generated-data support now exists; all blocks were unused at reservation. Keep final judging sealed until the candidate is fixed.
- Trainer production preparation: `trainer/cert/refit/jmix20.sh`, generated solver/protocolv3, `export_dials_s46_v2.py`; native converter `engineer/a52_pld/dly_from_trainer_v2.py`. Versioned originals preserved. Full16 exporter and sidecar roundtrip tested, including S46 late iso. Read current trainer note/receipts before launch because certification is still changing.
- Release now uses `release/release_gate_v2.py` to recompute the judge from exact paths/mode and check raw gates/hashes. Both fired S46 archives pass release-only checks and graded100/100. StageD was NOT RUN; `release/s46_worker_pair_v1.py` is prepared, previewed, and unlaunched pending a coordinated window. Exact current pair receipt: `O/lead/takeover_20261010T232330Z/graded_pair.json`.
- Next cheap hypothesis after cut-refit: separate EDGE bias/output per layer band8-9/10-12/13-14. Native source and tiny identity/accounting checks exist under `engineer/a56_edge_band/`; no fitted candidate or gain yet. Source-only cost: zero extra per-layer operations, six setup arrays for three bands.
- Public interface watcher and GPU truths remain live. No public designation opening or flopscope re-release found at last official check. Safari access was not approved; authenticated designation UI remains uninspected. No designation action taken.
- Six submissions were fired Oct10UTC; no Astra submission yet. Re-audit the UTC-day cap immediately before every fire. Goal remains first place, not just simulator completion.

## 14. Astra priority update 2026-10-11T00:09:05Z

Matt explicitly redirected focus to getting #1 and said not to worry about designation. Designation research is stopped; extra worker qualification of the already graded S46 pair is deferred. All new-gain release gates remain. Exact v11 simulator now passes forward fidelity on both 651 and378, removing about95% of original residual power; frozen profile/evidence in the latest TEAM_LOG receipt. Gradient/replay/FD certification remains in progress before cut-aware refit. No new candidate gain or submission yet.
