# Morning decision: Oct 16, for your 8:15 AM PT start

**Bottom line.** Nothing we have beats Oliver on the public board. The best candidate projects to 1.4724e-9 against Oliver's 1.4635e-9, so we stay 4th. Even so, two governed builds should be fired before the deadline (Oct 17, 4:59 PM PT) so they can serve as our private-run designations. The prize is decided on the private run, and our current public best (#336342) has no governor, which makes it unsafe there.

**Your decision:** your rule is "no submission until a stack beats Oliver". Lift it for exactly two fires: the primary (swap) and the secondary (S3). Written 10:20Z (3:20 AM PT); nothing has been fired.

## 1. Board (x1e-9, lower is better)
| marius | J2W | oliver | us (#336342) |
|---|---|---|---|
| 1.3551 | 1.4199 | 1.4635 | 1.4802 |

- Tying Oliver needs -1.13% vs 336342, or -1.28% with the 0.15% fire margin. Our best projection is -0.53%.
- The bill and wall side is exhausted: every remaining Smirnov trade buys about 0.03-0.06% of bill per projected second of wall.
- Accuracy lanes are still open (section 5).

## 2. Fire-ready candidates
All four are release-CONFIRMED, pass every gate, and are prefire CLEAR. None has been fired.

| Candidate | What it is | Projected public score | Gate P(loss)* | Platform risk** | Pre-registration (`lead/runs/opus_stack_v1/`) |
|---|---|---|---|---|---|
| **swap d81105ed** | final + wall-capped Smirnov read table, minus 12 read plans that save nothing | 1.4724e-9 (-0.53%) | 13.30% | like final | `PREREG_swap_v2.json` |
| **final 3786ca8b** | S2tr + governor v4g + push Smirnov + exact diets | 1.4738e-9 (-0.43%) | 13.13% | 0-15%, congested <= 57% | `PREREG_final_v4g_psm2_dm2_lb2.json` |
| **SAFE a4cee119** | final + request-margin fragments (+1.15% bill) | 1.4902e-9 (+0.67%) | 3.05% (first-priced) | 0-8% | `PREREG_safe_v2.json` |
| **S3 1e6536c5** | SAFE + earlier governor thresholds | 1.4910e-9 (+0.72%) | 3.05% (first-priced) | about 0-8% | `PREREG_s3safe_v2.json` |

- *Gate P(loss)* is the release judge's chance of losing a network, per 100 networks, with the governor switched off.
- *Platform risk* is u64's replay model (`lead/runs/opus_governor_v1/DESIGNATION_RECOMMENDATION_v1.md`): the chance a run loses at least one network, with the governor on.
- u64's 0%-risk "S3" also carried the read-plan prune (about -5 s of wall). Ours was built constants-only, so it sits between u64's S2 and S3.
- **swap vs final:** swap is -0.09% better on the public mix (SE about 0.06, from 10 fresh pairs). Its predictions match final's up to rounding, and its worst-case projected wall is 111.1 s against 111.8 s.

## 3. Designations
**How the private run scores:**
- Every network runs exactly once.
- A network that fails (residual over 0.4 s, or wall over 120 s) gets a zero prediction. That is about 4e7 times a normal network's error, so it ruins that submission's score.
- Only the better of our two designations counts.

**Why #336342 is out.** It has no governor. Our runs' residual tails have reached 0.41-0.48 s under platform congestion, and S2t lost networks to exactly these limits.

**Recommended pair: primary = swap (or final), secondary = S3.**
- When the primary runs clean, the secondary costs nothing.
- S3 fails under different conditions: about 7.6% fewer requests, and its governor steps in earlier. So it covers the cases that would sink the primary.

**Why both must be fired even if we stay 4th:**
- A designation can only point to a submission that exists on the platform.
- If we designate nothing, the default is our two best public scores. That would be #336342 (no governor) plus at most one governed build, and we would have no safe second.
- Public rank does not carry over. The private run re-scores the designated builds on networks nobody has seen, where leaders running near the limits can lose a network. J2W once lost 3 networks to timeouts.

## 4. Quota
- **Oct 16 UTC:** 7 of 10 used. 3 left until 00:00Z (5:00 PM PT today).
- **Oct 17 UTC:** 10 left. The deadline is 23:59Z (4:59 PM PT).
- **Suggestion:** fire swap and S3 today. If either loses a network publicly, there is time to re-fire on the 17th.

## 5. Still open overnight
- **[opus-ntsched]** (u64 + hunter): joins made earlier or extra using better-chosen subspaces. Its pre-registered test W13N, at 10:19Z, came out +1.41% error. That means the loss comes from the absorbed cohort itself, not the subspace, so the join-schedule family stays closed.
- **Still running at 10:20Z:**
  - u64's static twin, W13S;
  - the hunter's NTB32-L7 24-seed confirm. It passes only if the average per-network score change (xnet) is <= -0.1% with a negative aggregate, and the 16 A80 seeds are <= 0.
- **Anything else that lands before 15:00Z** will be in TEAM_LOG after 10:20Z. Read the `[opus]` lead lines first.

## Fire commands (the lead runs them; nothing is fired from this memo)
Run from `/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. These are verbatim copies of each `lead/runs/opus_release_v1/rel_<tag>/PIPELINE_DONE_v1.json`.
```
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py swap_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_swap_20261016_opus_v1/submission.tar.gz 'final_v4g_psm2_dm2_lb2 + wall-capped Smirnov read table (375 keys) - 12 zero-saving read plans (exact, rounding-level)'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py final_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_final_20261016_opus_v1/submission.tar.gz 'S2tr + governor v4g (residual and wall tiers) + push Smirnov v2 (wall-budgeted rule) + exact data-movement diets (dm2, lazy tree binds from setup boxes); FLOPs warm -0.22%.'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py safe_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_safe_20261016_opus_v1/submission.tar.gz 'SAFE: final stack (S2tr + governor v4g + push Smirnov v2 + dm2 + lazy binds) plus the two-strip covariance and first-five full-sandwich request margin; same model, rounding-level.'
/Users/matt/arc-white-box-estimation/.venv/bin/python -B /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py s3safe_20261016_opus_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_s3safe_20261016_opus_v1/submission.tar.gz 'S3 SAFE: the SAFE stack with a more conservative governor (residual tiers .37/.39/.43 s, wall tiers 108/115/121 s); same model, rounding-level.'
```
Each archive's sha256 matches its confirmation: swap 6d3612a4, final 2076e444, SAFE eaa57b1c, S3 d531d5cc.
