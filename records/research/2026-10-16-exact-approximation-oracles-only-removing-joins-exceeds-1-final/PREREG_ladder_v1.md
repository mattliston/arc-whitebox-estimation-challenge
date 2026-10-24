# [opus-hunteridea] PRE-REGISTRATION: oracle ladder on S2tr 46e9315d, then the cheap partial of the top approximation

**Written before any data.**

## Idea

S2tr still carries several engineering approximations around the physics. I want to know which single one now holds the most final-layer error. That question is untested on today's card: the oracles in memory come from older cards. I'll remove each approximation in isolation with a values-only switch and measure noise-free final-layer xMSE on 4 seeds against `ctl_s2tr`.

## Oracles

Each oracle is exactly one switch appended last to S2tr:

| Oracle | Switch | What it removes |
|---|---|---|
| O1 STAR | `_SS_RANK = (1024,)*15` | One-direction star (beta .91); gives exact rank-n star factors |
| O2 ULEG | `_LCR_RU = 384` | Pair-mean u-leg truncation at L10-14 (L15 stays 256) |
| O3 OTT | `_OTT_Z = _OTT_NEG_Z = 1e9` | Trace-only tail rows; every old-tier row is reconstructed in full |
| O4 DCV | `_DCV_FLOOR = 0.0` | Dead-unit covariance cut; full C at every layer |
| O5 RC | `_RC_COLZ = 1e9` | Read column-prefix cut; full live roster |
| O6 OLDTIER | `_JOIN_SCHEDULE = {}` | All Nystrom joins; every cohort stays exact. This is the joins lane (u64): diagnostic only, and dropped if the code path does not support it |

## Measure

- Seeds 7103248-51, screen_judge_opus_v1, against `ctl_s2tr`.
- Noise-free cross-half xMSE (per-seed log ratio, mean of 4), with the bill reported too.

## Expected effect

Prior: most of the remaining error sits in the modeled late moments (v, k3, k4), not in these approximations. So I expect each oracle to give at most about -1%. Any oracle at -2% or better would be a surprise worth chasing.

## Kill rule, stage 1

If no oracle reaches mean xMSE <= -1.0% on the 4 seeds, the approximation family is exhausted at a level that cannot pay 0.5%. Stop and report.

## Stage 2

Take the oracle with the most negative xMSE (<= -1.0%; O6 excluded, handed to u64). Build ONLY its pre-specified partial:

| Top oracle | Partial |
|---|---|
| STAR | Star rank 16 at all layers (`_SS_RANK = (16,)*15`) |
| ULEG | `_LCR_RU = 320` at L12-14 only, plus 384 at L10-11 (one table) |
| OTT | `_OTT_Z = _OTT_NEG_Z = 3.5` |
| DCV | `_DCV_FLOOR = 1e-4` |
| RC | `_RC_COLZ = 3.0` |

- Price it warm on 7103248.
- Screen it on 8 seeds (7103248-55) against `ctl_s2tr`.
- 8-seed xnet <= -0.15% goes to 24 seeds (`ctl_s2tr_a80` reused).
- Kill if the 8-seed xnet > 0 or the price exceeds +1.5%.

Bar for adoption: 24-seed xnet <= -0.3% with a negative aggregate.
