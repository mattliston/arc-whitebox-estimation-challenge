# HANDOFF TO ASTRA (interim LEAD), current as of 2026-10-13 19:00Z (12:00 PDT)

Written by the Opus lead, who is at its usage limit. Earlier layered versions: HANDOFF_TO_ASTRA_20261013T1625Z.md(.v1/.v2_layered). THIS file is the one to follow.

**Paths:**
- O = /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
- TEAM_LOG = O/TEAM_LOG.md (append-only; every stamp from `date -u` in the same command)
- Ledger = /Users/matt/arc-white-box-estimation/autoresearch/SUBMISSION_LEDGER.md
- Lead tools = O/lead/tools: grade_vs.py, lead_watch_v4.sh

## 0. Goal and honest position

**Goal:** public top 3 by 2026-10-17 23:59 UTC. Score = final-layer MSE x max(FLOPs/2^41, 0.1), mean over the 50 public networks.

| Place | Team | Score |
|---|---|---|
| 1 | marius | 1.3882e-9 |
| 2 | J2W | 1.4347e-9 |
| 3 | oliver | 1.4989e-9 |
| 4 | Luna | 1.5199e-9 |
| 5 | **us** | **1.52120e-9 (#335782)** |

- Luna is +0.09% ahead of us; oliver is -1.46% away and improving.
- Passing Luna is within reach today. Third needs about 1.5-2% more, so stack EVERY validated small gain and keep hunting for bigger ones.

**Designation pair:**
- #335782 (primary): s54k1 twin = S54 + carried deltas + K4L-inner (K4L ages 3-4 at 0.5 at L8-14).
- #335822 (s54k1 SAFE, same physics): 1.52412e-9, 0 failed.

**Fires used:** 5 of 10 on 10-13 UTC. Today's last fire must go in before 00:00Z (17:00 PDT); after that the count resets to 10.

## 1. Live pipeline (all automated; the lead's jobs are decision files and firing)

**A. S56 → s56k1 (the main fire tonight)**
- S56 is the RAD round with K4L-inner absorbed into the RAD card. Sizing was 435 of 512 at 19:00Z, so it ends about 19:40Z. Then come the check (about 30 min) and engineer2's port (loop v17, about 1.5 h): FILES READY about 21:30-22:00Z.
- scout-native's chain_round_stack_v7 (PID 1651) builds s56k1 automatically. It reads O/scout_native/S56_DECISION, already written as `fired s54k1tw / extra none`.
- **Just before its build** it reads O/scout_native/S56_EXTRA_AMEND if that exists. The amend's first line = extra fragment names, which REPLACE the extra list. Write the amend only for a validated gain; never overwrite S56_DECISION.
- Then: census → prebind → 7100136 identity → counts → release trigger → seal on 7190032-039 (base side already done) vs the fired s54k1tw, default rule.
- **FIRE iff all three hold:**
  - STACK s56k1 SEALED ends "-> FIRE";
  - `[release] ... TWIN s56k1tw CONFIRMED` (it contains the exact fire command);
  - the prefire check is CLEAR (section 2).
- Expected fire about 23:00-00:00Z (16:00-17:00 PDT). If it passes after 00:00Z, it uses 10-14's quota.

**B. Candidate extras for s56k1 (write S56_EXTRA_AMEND only when validated, before s56k1's build)**
- **K31L4 age 4 (T147 h): PASSED at 24** on top of C: -0.157% (se 0.049, 17 of 24), out-of-selection -0.115%.
  - Native form = band L11-14 (fragment_k31a4_b1114_v1, being built by scout-native), bill about +0.04%.
  - scout-native is measuring the native ten vs quick_s54k1tw_a.json at 19:0xZ.
  - If its net (MSE + bill) on the ten is negative, write S56_EXTRA_AMEND = `fragment_k31a4_b1114_v1.py`.
- **T151 carrier-dial scan** (physicist; screen on FRESH 7100608-623, confirmation on 7100592-607; pass = all-32 <= -0.15% and confirmation sign agrees):
  - arms: law plane strength nfs +/-0.15 at L7-14, law dscale nfd +/-0.15 at L1-14, K31L4 plane x1.2 at L7-14;
  - screen about 21:10Z, confirmation about 21:30Z;
  - a winner needs a native literal fragment with parity D >= 0.5 from scout-native / engineer2 before the amend.
- **T150 four-form (result 19:12Z): not actionable as run.**
  - Under the all-row fit, dfit 2 is +10.1% vs C. Against that dfit 2, dfit 3 (the constant regressor) is -3.46% and dfit 4 is -4.2%, 8 of 8 each, but both stay about +6% worse than C.
  - So the lead cancelled the all-row extension.
  - The physicist checks whether adding the constant regressor to the RESPONSE-row fit (what C and native use) helps. If so, that's T153: a fresh-network screen vs C. Its native port should be cheap.
  - The C^2 part adds only about -0.7% and isn't worth its bill.
- **T152 flips (physicist, pre-declared 19:02:28Z):** (m) pbase so2m 1.0 at L1-14; (x) x-half (x1 1.6, x2 5.0, via remainder31).
  - Fresh protocol: screen on 7100608-623, confirm on 7100592-607.
  - Screen decision about 21:45-22:15Z. A smoke test runs first.
  - so2m is cheap natively (elementwise). x-half needs a separate read, so it is port-worthy only if large.
- **scout-card K25:** the L10 join rank 384 re-run on a private cutsim copy (RANKS[10] = 384). The bill is +1.65% natively, so it needs an MSE gain clearly above that to net.

**C. If a validated gain lands AFTER s56k1 is built:** do a quick fired-file stack on the newest fired twin. Ask scout-native; chain_fired_stack_v1.sh with a decision file is the pattern.

## 2. How to fire

1. **Prefire:**
   `cd O && /Users/matt/arc-white-box-estimation/.venv/bin/python -B fire/prefire_check_v1.py <abs archive> <sha8> <abs archive of the last fired TWIN>`
   - The last fired twin is O/release/runs/a_s54k1tw_safe/submission.tar.gz.
   - It must print `PREFIRE CLEAR | new signatures: 0`.
   - Imports must be only base64 / flopscope / flopscope.numpy / math / time / whestbench / zlib.
2. **Fire:** release's FIRE COMMAND verbatim: `/Users/matt/arc-white-box-estimation/.venv/bin/python -B fire/fire_once.py <tag> <abs archive> '<desc>'` from O. Tag style is s56k1twin_20261013; it prints the submission_id.
3. **Poll:** `cd O && nohup bash fire/poll_eval.sh <tag> <id> 150 > fire/<tag>/poll.out 2>&1 &`. It's done when fire/<tag>/poll.log shows `completed 100 0`, about 50 min.
4. **Grade:** `python3 O/lead/tools/grade_vs.py O/fire/<tag>/eval_<id>.json O/fire/s54k1twin_20261013/eval_335782.json` (paired, on the public 50).
5. **Log:** ledger FIRE and GRADED lines in the existing style, plus TEAM_LOG lines starting `[astra] <date -u stamp>`.
6. **After a fire that changes the physics:** tell the physicist to update physicist/t127_baseline_carry.json to the fired form. Tell the trainer which fired deltas to absorb (cert/RAD/ABSORB.json, absorb_v3) from the next round whose config hasn't run.
7. **Designation:** after a new best twin, ask release for a same-physics SAFE (as s54k1s1 was built; release/safe_designation_v1.sh), fire it once gated, and update the pair.

## 3. Hard rules

- **NEVER** read, print, copy or cat key.txt or any credential.
- Only fire/fire_once.py submits. At most 10 submissions per UTC day.
- Fire only verified improvements: sealed < 0, release CONFIRMED, prefire clear. A SAFE goes out only as the designation fallback.
- No AIcrowd forum reading.
- 48 GB Mac on a 96 W adapter, memory-bound under load.
  - Keep about 10 or fewer heavy processes. Gate new compute on memory_pressure free >= 50%.
  - nice 10, BLAS 1, guard_footprint_v3 (--rss-backstop 7500000000 for native).
  - Never kill other lanes' processes; stop only your own exact PIDs.
- Never edit files that running processes import, and never overwrite decision files: version them (_v2, _AMEND).
- Never quote other lanes' trigger strings inside your TEAM_LOG lines; their watchers match unanchored.
- **Times to Matt in PT** (PDT = UTC-7). Report every grade as the score, the paired change vs the best, and the gaps to Luna and oliver.
- **Screens:** 8-network screens on 7100584-591 have given repeated FALSE POSITIVES today (c0, pe3 died out of sample). Decide only on 24+ networks with the out-of-selection sign rule, and prefer fresh networks.

## 4. Closed today (don't redo)

- CFCOV +0.20%.
- K19 L15 Edgeworth (both signs).
- T140 / T144: K4L ages 5-6.
- K22: E33 +2.1%, F4C S1 null.
- K23: F4C age 3 half strength, K31L age 4 at L8-11.
- T142: C's strength is optimal.
- T143: no per-unit k4 source is left.
- T145: K4L age-2 x0.75 / x1.25.
- K24: K31L4 x0.75, F4C x0.85, K4L 1-2 x0.9.
- T146: c0 +0.05 / 0.10 / 0.15 failed out of sample. RAD gradients are biased for carrier dials.
- T148: pe3 at L8-11 failed out of sample.
- r_u 320 null.
- T147 (i): the age-3 → age-4 redistribution is null.
- S55: not accepted.

## 5. Monitoring

- `tail -n 40 O/TEAM_LOG.md`.
- `bash O/lead/tools/lead_watch_v4.sh [fire_tag]` exits on the next trigger (state in the scratchpad's waiter_state; edit the path if needed).
- Lanes (Claude sessions; they may be paused by the shared usage limit, but their background chains run):
  - trainer: RAD rounds S56-S66 unattended;
  - engineer2: port loop v17;
  - scout-native: stack chains;
  - release: gates;
  - physicist: screens, stage 2;
  - scout-card: screens;
  - analyst: truths, board watch.
