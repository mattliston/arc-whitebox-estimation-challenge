# PATH TO 1.5e-9 — radlab ledger v1 (no compute)

Written by radlab at about 22:58Z on 2026-10-12 and saved by the lead, because the harness blocks radlab's .md writes. Text as received.

**Headline:** the central path ends at about 1.526e-9 (x 0.988). That is 1.7% short of 1.5e-9; from 1.54442e-9 we need -2.88%. Even with every item at its high end we only reach about 1.502e-9.
- The listed items alone give under 5% chance of a public score below 1.5e-9.
- They do roughly double the podium odds: about 17% now, about 34% at x 0.988 on the analyst's curve.

## Conventions
- Board effect = 0.8 x raw torch for card / error items; 1:1 for byte-identical bill items.
- E = probability x effect.
- Board se is about 0.10% per paired comparison.

## Items

| item | board effect (central; range) | P | E | ETA |
|---|---|---|---|---|
| #335633 first stack (S51 + gk + SN3) | partial -0.30% at 25 / 50 (parts predict -0.40..-0.50: gk 0.8 x -0.35 = -0.28 + SN3 bill -0.11..-0.23) | 1 (fired) | -0.30 (update when final) | 23:10Z |
| s51k2: rows2 | -0.08 bill | 0.85 | -0.07 | stack 2 |
| s51k2: npx1 | -0.07 | 0.8 | -0.06 | stack 2 |
| s51k2: gnnw2 | -0.095 bill (zero-mean MSE, adds only per-network noise) | 0.85 | -0.08 | stack 2 |
| s51k2: a4k31 FULL | 0.8 x -0.171 + 0.12 bill = -0.02, about 0 | | | |
| s51k2: a4k31 BAND (L12-15) | its gain is 83% born at L12-15, so it keeps about 80%: -0.11 + 0.033 = -0.08 (L13-15: -0.09 + 0.023 = -0.07) | 0.6 | -0.05 | stack 2 / 3 |
| K7 / K8 winners (lq, gbd2, gk15 null) | K8 gks / bsksplit / bsklate: -0.1..-0.2 if one is real (bsk's late part is about -0.24 raw once the L6-12 harm is gone) | 0.35 | -0.05 | 00:40Z + 24-net confirm |
| K7 rest (gbeta / gomit / gbu / bk4l) | | 0.15 | -0.015 | |
| engineer2 list D bill cuts (sites 1.25%) | recoverable fraction unknown; central 25% = -0.31 (0..-0.6) | 0.7 | -0.22 | unknown (engineering hours) |
| scout-native TRX | 0..-0.3, central -0.12 | | -0.10 | ? |
| S53 | -0.07 (+-0.10) | 0.9 | -0.065 | about 05:00Z |
| S54+ lean rounds to 10-17 | -0.05..-0.15 total (supply-limited) | | -0.08 | |
| physicist T132 | at most -0.4 | 0.2 | -0.08 | ? |

## Sum
- Items after #335633: E about -0.87%.
- With #335633: -1.18%, so 1.54442e-9 x 0.9882 = 1.5262e-9.
- If #335633 finals at -0.45: about 1.524e-9.
- Spread of the sum: sd about 0.34% (independent items; list D, T132 and the board noise dominate).
- All items at their high end, all succeeding: -2.75%, about 1.502e-9.
- SHORTFALL at the central path: 1.7% beyond everything listed.

## Where an extra hour or a lane raises the expected total most
1. **engineer2's list D.** It has the largest E (-0.22) and the widest range (0..-0.6), and it transfers 1:1 with no torch risk.
   - Pricing each site's recoverable fraction first is the most valuable hour on the board: it turns 0..-0.6 into a number.
   - A second engineer-type lane on it is the best extra lane.
2. **Cheap torch screens of L12-15 knobs on top of gk.**
   - After gk, 2/3 of the final error is born at L12-15.
   - An 8-network arm costs about 0.5 worker-hours, with P(win) about 0.3 and a payoff of 0.1-0.2 on the board, i.e. about 0.1%/worker-hour of expected value. RAD sizing gives about 0.02%/h, five times less.
   - Concretely: the a4k31 BAND as a K-arm now (the full form nets about 0 after its bill), and K8's bsksplit / gks to 24 networks if they show.
   - If you must choose, move one RAD worker to the scout queue.
3. **The L8-11 band.** It is about 1/4 of the final error, and nothing with the right sign moves it except a4k31s6 (-0.13 at L9 / L11).
   - gbeta1 moved L5-9 by +0.5..+1.1% on 2 networks (wrong sign).
   - A sign / strength scan there is the only route I see to an error gain larger than about 0.3%.

**Honest bottom line:** the live list is worth about -0.9% (plus the first stack's -0.3..-0.5), not -2.9%. Closing to 1.5e-9 needs about 1.7% from items not yet on the list. The cheapest place to look is bill (list D at full recovery). After that comes a mid-band (L8-11) error lever, which nothing found so far has touched.

## Lead's notes (22:59Z)
- s51k2 was re-scoped at 22:57Z to the byte-identical items only (rows2 + npx1). GNN-W is parked: scout-native corrected its MSE effect, because a coarser kernel adds error, putting its net at -0.035..-0.095%. Full-layer a4k31 is out.
