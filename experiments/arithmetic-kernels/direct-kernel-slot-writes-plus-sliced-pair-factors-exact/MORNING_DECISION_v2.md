# Morning decision v2: Oct 16, for your 8:15 AM PT start

This replaces v1. Our public best is #336412, not #336342: the board scores a run whose 50 public networks all complete, even if hidden networks fail.

**Bottom line.** We are 4th at 1.475902e-9 (#336412 = S2tr). Nothing we have reaches Oliver: that needs -0.84% from here, or -0.99% with the fire margin. Our best candidate, the swap, projects to -0.24% (1.4723e-9). That would be a new best for us, but still 4th. Separately, two governed builds must be fired before the deadline (Oct 17, 4:59 PM PT) to serve as private-run designations. Both our public bests (#336412 and #336342) lack a governor, and #336412 already lost a hidden network (MLP93) on its own run.

**Your decision:** lift "no submission until a stack beats Oliver" for exactly two fires, swap (primary) and S3 (secondary). Updated 10:30Z (3:30 AM PT); nothing has been fired.

## 1. Board (x1e-9, lower is better)
| marius | J2W | oliver | us (#336412) |
|---|---|---|---|
| 1.3551 | 1.4199 | 1.4635 | 1.4759 |

- Gap to Oliver: -0.838% (-0.987% with the 0.15% fire margin).
- **Bill and wall side:** exhausted. Every remaining Smirnov trade buys about 0.03-0.06% of bill per projected second of wall.
- **Overnight accuracy lines:** all closed. The join schedule and NT-joins families failed their pre-registered tests at 10:24-10:25Z.

## 2. Fire-ready candidates
All four are release-CONFIRMED, pass every gate, and are prefire CLEAR. None has been fired.

| Candidate | What it is | Projected public (vs 336412) | Board risk: a public net fails* | Private-run risk: any net fails** |
|---|---|---|---|---|
| **swap d81105ed** | final + wall-capped Smirnov read table, minus 12 read plans that save nothing | 1.4723e-9 (-0.24%) | 7% gate; 0-8% governed (34% congested) | 13% gate; 0-15% governed (57% congested) |
| **final 3786ca8b** | S2tr + governor v4g + push Smirnov + exact diets | 1.4737e-9 (-0.15%) | as swap | as swap |
| **SAFE a4cee119** | final + request-margin fragments (+1.15% bill) | 1.4901e-9 (+0.96%) | 1.5% gate; 0-4% governed | 3% gate; 0-8% governed |
| **S3 1e6536c5** | SAFE + earlier governor thresholds | 1.4908e-9 (+1.01%) | 1.5% gate; under 4% governed | 3% gate; under 8% governed |

- **How the projection is built:** #336412 is S2tr itself, so each candidate is 336412 times (1 + its bill change vs S2tr + its expected governor cost). All the code changes are exact up to rounding.
- **Pre-registrations** (in `lead/runs/opus_stack_v1/`): `PREREG_swap_v2.json`, `PREREG_final_v4g_psm2_dm2_lb2.json`, `PREREG_safe_v2.json` and `PREREG_s3safe_v2.json`. These are written against #336342. `PREREG_rebase_336412_v1.json` restates them against #336412.
- *Board risk* is the chance that one of the 50 public networks fails, which wipes that run from the board. It is about half of the private-run risk: 1 - sqrt(1 - P(any of 100)).
- **"Gate"** is the release judge's figure, per 100 networks, with the governor switched off.
- **"Governed"** is u64's platform replay with the governor on (`lead/runs/opus_governor_v1/DESIGNATION_RECOMMENDATION_v1.md`). The range runs from early to late detection of a slow run.
- **S3 caveat:** u64's 0%-risk S3 also carried the read-plan prune. Ours was built constants-only, so it sits between u64's SAFE row and that S3 row.
- **Private-run risk is quoted per 100 networks.** A larger private suite raises it.

## 3. Designations
**How the private run scores:**
- Every network runs exactly once.
- A failure (residual over 0.4 s, or wall over 120 s) gives a zero prediction. That is about 4e7 times a normal network's error, so that submission is ruined.
- Only the better of our two designations counts.

**Our ungoverned builds should not be designated.** The default designation is our two best public scores, today #336412 and #336342. Neither has a governor, and #336412 already lost a hidden network (MLP93) on its own run; on the private run that loss would have counted.

**Recommended pair (unchanged): primary = swap (or final), secondary = S3.**
- When the primary runs clean, the secondary costs nothing.
- S3 fails under different conditions: about 7.6% fewer requests, and its governor steps in earlier. So it covers the cases that sink the primary.

**Why both must be fired even if we stay 4th:**
- A designation can only point to a submission that exists on the platform.
- A clean swap fire also becomes our new public best, about 93% likely by the gate model.
- S3 scores worse publicly, so it will never enter the default pair. It has to be fired and designated explicitly.
- Public rank does not carry over. The private run re-scores the designated builds on networks nobody has seen, and leaders running near the limits can lose a network there.

## 4. Quota
- **Oct 16 UTC:** 7 of 10 used. 3 left until 00:00Z (5:00 PM PT today).
- **Oct 17 UTC:** 10 left. The deadline is 23:59Z (4:59 PM PT).
- **Suggestion:** fire swap and S3 today. If a public network fails, there is time to re-fire on the 17th.

## 5. Overnight lines
- **[opus-ntsched] is CLOSED.** This line tested making joins earlier or extra, using better-chosen subspaces. The loss is cohort content, not subspace: W13S (static) came out +2.12% error, W13N (near-oracle subspace) +1.41%.
- **NTB32-L7 FAILED** its 24-seed confirm (u64's NT-joins arm, checked by the hunter): the average per-network score change (xnet) was -0.04% against a pass bar of -0.1%, and the independent A80 seeds gave +0.02% where the pass needed 0 or better. The NT family is closed.
- **Anything that lands before 15:00Z** will be in TEAM_LOG after 10:30Z. Read the `[opus]` lead lines first.

## Fire commands (the lead runs them; nothing is fired from this memo)
Unchanged from v1, and verbatim in each `lead/runs/opus_release_v1/rel_<tag>/PIPELINE_DONE_v1.json`. Run from `/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`:
```
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py swap_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_swap_20261016_opus_v1/submission.tar.gz 'final_v4g_psm2_dm2_lb2 + wall-capped Smirnov read table (375 keys) - 12 zero-saving read plans (exact, rounding-level)'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py final_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_final_20261016_opus_v1/submission.tar.gz 'S2tr + governor v4g (residual and wall tiers) + push Smirnov v2 (wall-budgeted rule) + exact data-movement diets (dm2, lazy tree binds from setup boxes); FLOPs warm -0.22%.'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py safe_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_safe_20261016_opus_v1/submission.tar.gz 'SAFE: final stack (S2tr + governor v4g + push Smirnov v2 + dm2 + lazy binds) plus the two-strip covariance and first-five full-sandwich request margin; same model, rounding-level.'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py s3safe_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_s3safe_20261016_opus_v1/submission.tar.gz 'S3 SAFE: the SAFE stack with a more conservative governor (residual tiers .37/.39/.43 s, wall tiers 108/115/121 s); same model, rounding-level.'
```
Each archive's sha256 matches its confirmation: swap 6d3612a4, final 2076e444, SAFE eaa57b1c, S3 d531d5cc.
