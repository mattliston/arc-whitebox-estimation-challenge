# RADLAB REPORT v1 (radlab, sent ~16:4xZ 2026-10-12; saved by the lead at 2026-10-12T16:40:43Z because the harness refused radlab's own write)

## Bottom line
1. Re-sizing pays ~0. Fresh A41 at the S50 point, 9 of S50's strength arms re-sized: chi2 6.2 on 9 df; every single-arm LOO >= -0.001%; joint LOO > 0 at every ridge.
2. Gains come from directions sized for the FIRST time. S50: LOO -0.226%, fresh -0.26% on 96, board -0.244%. S51's new arms on fresh A41: LOO -0.099% (se 0.063): arrp -0.046, dk21n -0.031, drsy -0.022 (chi2 19.6 on 4 df).
3. Hypothesis: new signal appears when the CARD changes (Ne/Nf re-derived after Nd froze NULL; arrp paid again after the S48 patch). If true, a RAD round is worth most right after a structural change: turn rule (c) ON once there.
4. Supply: about 0.1-0.3% (central 0.2%) left in the RAD line after S51, banked within 3-6 accepted lean rounds; not 16 x 0.2%.
5. The -SE acceptance rule is now the biggest cost. P(pass | true gain 0.05 / 0.10 / 0.20%) = 31 / 50 / 84% at check SE 0.10%. LOO predicts fresh gain well at n >= 32 (Nd -0.46 -> -0.48; MS -0.08 -> -0.22; S50 -0.23 -> -0.26; S48 at n=16 missed). Proposal: continue unless the pooled check is worse than +1 SE (or combined LOO+check z < -1). Firing still needs stage 2 + release against the last FIRED file.
6. The strongest missing directions need native ROUTES: single-layer signals bscale L0 (t -5.26), k4 L15 (+4.77), k4 L7 (+3.06), k22 L14 (+3.40), x2a1 L7 (-3.27), X1 L14 (-2.85). Expected fresh t 1.5-2.8, about -0.01..-0.06% each, once.
7. No help: pruning inside a round (error-neutral; saves hours only), loosening rule (b) (A36 t overstates fresh t ~2x: fresh t = 0.54 x A36 t, corr 0.89), 64-network rounds (eta 0.84 -> 0.69), phase 2 (+0.01% for ~4 h), a repair arm (~0).

## Step efficiency
eta(n) = 1 - 20/n from S50's learning curve (held-out -0.081 / -0.135 / -0.210 / -0.23..-0.26% at 16 / 32 / 48 / 64 sizing networks; G_inf ~0.36%).

## Design scenarios (radlab/design_v1.py; signal model = first lean round harvests s0, an accepted round leaves rho, a rejected one uses nothing)
| design | h/round | rounds | accepted | total (s0 .05, rho .5) / (.05, .8) / (.10, .7) |
|---|---|---|---|---|
| (a) current 14 arms, -SE rule | 8.9 | 12 | 3-4 | -0.066 / -0.097 / -0.198 |
| (a2) current, relaxed rule | 8.9 | 11 | 10 | -0.084 / -0.185 / -0.272 |
| (b) rule (c) off, escal out, -SE | 4.6 | 21 | 4-6 | -0.078 / -0.137 / -0.237 |
| (b2) (b) + relaxed rule | 4.6 | 18 | 15 | -0.084 / -0.204 / -0.280 |
| (c) (b2) at n 64 | 2.8 | 25 | 22 | -0.069 / -0.170 / -0.229 |
| (d) (b2) + 6 weak bands | 7.8 | 12 | 11 | -0.101 / -0.207 / -0.291 |
| (d2) (b2) + one aggregate weak-band arm | 5.2 | 16 | 14 | -0.097 / -0.215 / -0.292 |

## Still open (radlab's recommendations)
1. Acceptance: continue unless the check is worse than +1 SE (before S52's check).
2. Run stage 2 + port + release only when the cumulative LOO since the last fired file reaches about -0.08% (fewer 2-worker holds).
3. Rule (c) ON once in the first round after an accepted structural change.
4. Each certified single-layer route goes into rule (b) as its own band.
5. Stop/slow: two accepted rounds in a row with LOO above -0.03% -> line to 2 workers, CPU to error-side work.
6. Optional: rotate the gradient networks (the previous round's reserve block instead of A36).
7. The two-stage prune is error-neutral (S50 cross-fit +0.003 ± 0.013%): fine to wire.

Pre-registered for S52 v2: LOO -0.02..-0.05%; true fresh -0.01..-0.06%; P(accept) 25-40% under -SE, about 90% under +1 SE.

Data: O/radlab/yield_S50_v1.json, yield_S51_A39p13_v1.json, yield_S51_A41p13_v1.json, grads_S48_v1.txt, grads_S51_v1.txt; scripts O/radlab/{build,yield,crossfit,curve,grads,design}_v1.py.
