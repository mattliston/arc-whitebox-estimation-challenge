# Morning decision v3: Oct 16, for your 8:15 AM PT start

This replaces v2. It adds public-only builds, which are for the board and never for designation, and an updated board.

**Bottom line.** We are 4th at 1.475902e-9 (#336412 = S2tr). Nothing we have reaches Oliver: that needs -0.84% from here, or -0.99% with the fire margin.
- **Best public-only build:** pubmaxapg2 projects to -0.42% (1.4697e-9) on a clean run. That would be a new best for us, still 4th.
- **Designations:** two governed builds still have to be fired before the deadline (Oct 17, 4:59 PM PT). These are swap (primary) and S3 (secondary). Both our public bests lack a governor, and #336412 already lost a hidden network (MLP93).

**Your decisions:**
1. Lift "no submission until a stack beats Oliver" for the two designation fires: swap and S3.
2. Optionally, allow one public-only fire for the board: pubmaxapg2.

Updated 11:25Z (4:25 AM PT); nothing has been fired.

## 1. Board (public scan 11:04Z, x1e-9, lower is better)
| marius_binner | J2W | oliver_chen | us (#336412) | Luna |
|---|---|---|---|---|
| 1.3519 | 1.4199 | 1.4635 | 1.4759 | 1.4843 |

- marius improved from 1.3551 (#336419, 05:34Z).
- Luna is 0.57% behind us.
- A board scan runs hourly. Changes are logged in TEAM_LOG as `[opus-pubgov] BOARD CHANGE`.

## 2. Designation candidates (unchanged from v2)
| Candidate | Projected public (vs 336412) | Board risk: a public net fails | Private-run risk: any net fails |
|---|---|---|---|
| **swap d81105ed** (primary) | 1.4723e-9 (-0.24%) | 7% gate; 0-8% governed | 13% gate; 0-15% governed |
| final 3786ca8b | 1.4737e-9 (-0.15%) | as swap | as swap |
| SAFE a4cee119 | 1.4901e-9 (+0.96%) | 1.5% gate | 3% gate; 0-8% governed |
| **S3 1e6536c5** (secondary) | 1.4908e-9 (+1.01%) | 1.5% gate | 3% gate; under 8% governed |

**Designation logic:**
- Every network runs once on the private run.
- A failure gives a zero prediction, about 4e7 times a normal network's error.
- Only the better of the two designations counts.
- The default pair (#336412 + #336342) has no governor.
- The pre-registrations are `PREREG_swap_v2`, `PREREG_final_v4g_psm2_dm2_lb2`, `PREREG_safe_v2` and `PREREG_s3safe_v2`, plus `PREREG_rebase_336412_v1`, all in `lead/runs/opus_stack_v1/`.

## 3. Public-only builds (board fires only; NOT designation candidates)
A public fire only needs its 50 public networks to finish. A hidden-network failure costs nothing, and a run that ends worse than #336412 changes nothing.
- **What counts against a fire:** a LEAN or SHED trigger on a public network (+36% error on one network, +0.72% on the run, or worse) wipes the gain, just as a failure does.
- **What counts for it:** only the bill-only governor tiers. FULL (0.39 s) and EXACT (0.41 s / 112 s wall) prevent failures for about 1.65% of one network's bill each.

The table below uses u64's refined replay model ("post" mode).
- **Gain per fire** is the expected board improvement. The first figure is for normal platform conditions; the second weights in 25% congested.
- **Clean run** means all 50 public networks complete with no LEAN or SHED trigger.

| Build | What changes | Gain per fire | Clean run vs 336412 | P(public) / P(any) at L3 / L6 / L8 |
|---|---|---|---|---|
| swap d81105ed (v4g) | reference | 0.233 / 0.193% | -0.26% (1.4721e-9) | 0/0, 4/8, 8/15% |
| swappub eafb6341 | swap with public-fire thresholds | 0.262 / 0.240% | -0.29% (1.4716e-9) | 0/0, 4/8, 8/15% |
| pubmaxa 0bc7bd5d (u64) | swap + uncapped read grid; LEAN > 122 s, SHED > 125 s | 0.334 / 0.273% | -0.38% (1.4704e-9) | 4/8, 8/15, 12/22% |
| **pubmaxapg2 0353ac45** | pubmaxa + residual tiers 0.40/0.42 s and EXACT wall 116 s | **0.375 / 0.339%** | **-0.42% (1.4697e-9)** | 4/8, 8/15, 12/22% |
| pubmaxapg c31ca568 | pubmaxapg2 with LEAN-wall off, SHED 123 s | 0.375 / 0.339% | -0.42% | 0/0, 4/8, 12/22% |

**Recommended public-only build: pubmaxapg2.**
- It beats u64's pubmaxa on every measure: +0.04 to +0.07% gain per fire, -0.42% vs -0.38% clean run, and -0.16% vs -0.12% on u64's own linear net.
- **pubmaxapg** scores the same on the board measure. On u64's linear measure it shows a +3% spike, from SHED on the record's one 132.6 s timeout network. That spike does not matter for the board, but pg2 avoids it.
- **swappub** is the swap's public-fire twin, if you would rather not fire the uncapped read grid.
- **All three of my builds are constants-only** on their base, with parity copies identical to A's (42c60af4) or the swap's (bc189784). All are release-CONFIRMED and prefire CLEAR.
- **Caveat:** the replay counts are small (a few LEAN networks per 1,200). The structural part is robust: LEAN and SHED never help a public fire.
- **Pre-registration:** `lead/runs/opus_stack_v1/PREREG_public_fires_v1.json`. It also contains the raw-mode numbers and the congested cases.

## 4. Quota
- **Oct 16 UTC:** 7 of 10 used. 3 left until 00:00Z (5:00 PM PT).
- **Oct 17 UTC:** 10 left. The deadline is 23:59Z (4:59 PM PT).
- **Suggestion:** fire swap + S3 for the designations, and optionally pubmaxapg2 for the board. That uses today's three slots and leaves 10 for re-fires on the 17th.

## 5. Overnight lines
- **[opus-ntsched]:** CLOSED. The loss came from the absorbed cohort itself, not the subspace.
- **NTB32-L7:** FAILED its 24-seed confirm.
- **[opus-pubmax]:** the uncapped grid and join/old-bank Smirnov do not pay under v4g. The grid pays only with later LEAN/SHED, which is pubmaxa / pg2.
- **[opus-pubgov]:** the public-fire thresholds above. The hourly board watch is running.
- **Anything after 11:25Z:** in TEAM_LOG.

## Fire commands (the lead runs them; nothing is fired from this memo)
Run from `/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. These are verbatim copies of each `lead/runs/opus_release_v1/rel_<tag>/PIPELINE_DONE_v1.json`. Archive sha256s match their confirmations:

| Build | Archive | Confirmation |
|---|---|---|
| swap | 6d3612a4 | 9ed43e63 |
| final | 2076e444 | a68e348a |
| SAFE | eaa57b1c | a717694b |
| S3 | d531d5cc | afd6b35f |
| swappub | d76bbaf4 | f72d5d30 |
| pubmaxa | eb4b9d33 | 34765ca2 |
| pubmaxapg | bf718b3e | 2ca176fd |
| pubmaxapg2 | 7b1971be | 23e652c3 |

```
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py swap_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_swap_20261016_opus_v1/submission.tar.gz 'final_v4g_psm2_dm2_lb2 + wall-capped Smirnov read table (375 keys) - 12 zero-saving read plans (exact, rounding-level)'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py s3safe_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_s3safe_20261016_opus_v1/submission.tar.gz 'S3 SAFE: the SAFE stack with a more conservative governor (residual tiers .37/.39/.43 s, wall tiers 108/115/121 s); same model, rounding-level.'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py pubmaxapg2_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_pubmaxapg2_20261016_opus_v1/submission.tar.gz 'PUBLIC-ONLY max, robust: swap + uncapped Smirnov read grid, governor residual 0.40/0.42/0.46 s, wall 116 s exact, LEAN above 122 s, SHED above 125 s; same model, rounding-level; not a designation candidate'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py pubmaxa_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_pubmaxa_20261016_opus_v1/submission.tar.gz 'PUBLIC-ONLY max: swap + uncapped Smirnov read grid, governor v4h (LEAN above 122 s, SHED above 125 s); same model, rounding-level; not a designation candidate'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py swappub_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_swappub_20261016_opus_v1/submission.tar.gz 'PUBLIC-FIRE swap: swap d81105ed with governor thresholds for public fires (residual 0.40/0.42/0.46 s, wall 116 s exact, LEAN-wall off, shed 123 s); same model, rounding-level'
```
The final, SAFE and pubmaxapg commands are in their `PIPELINE_DONE_v1.json` files, same pattern; see MORNING_DECISION_v2.md for the first two.
