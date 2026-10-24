# Frame quality per join x fewer joins (fable_physicist_v1/frame, 2026-09-22 night)

Harness `frame_join_combo.py`: the lead's join-schedule push with five frame methods at each join, applied to the implicit weighted
Gram G = J diag(lam^2) J^T + Qp core Qp^T:
- sketch = the bank's one pass qr(G W[:, :384])
- power1 = one power step
- over448 = 448-wide sketch then Rayleigh-Ritz to 384
- warm = sketch from [Qp, 64 W columns] then Rayleigh-Ritz
- eig = exact top-384 eigenvectors of the dense G (the ceiling)

Parity: every/sketch = bank bit-exact. Torch, learned ON, no refit.

## Result: the schedule matters, the frame does not

Raw change vs bank on 35 / 100 / 37 / 39 (`fj_grid4.jsonl`):

| schedule | sketch | power1 | over448 | warm | eig |
|---|---|---|---|---|---|
| every (bank) | 0 | -0.67% | -0.35% | -0.14% | -0.10% |
| even | -0.91% | -1.25% | -1.67% | -1.57% | -1.54% |
| every3 (L7/10/13) | -1.86% (3/4) | **-2.04% (4/4)** | -1.85% (4/4) | -1.73% | -1.62% |
| every3b (L8/11/14) | -1.42% | -1.39% | -1.40% | -1.44% | -1.37% |

TRAIN12 (36/38/40/41/43/44/45/46/47/49/50/51; `fj_train12.jsonl`):

| combination | mean | wins | worst net |
|---|---|---|---|
| every3b / sketch | -2.87% | 10/12 | +0.34% |
| every3b / power1 | -2.87% | 10/12 | +0.37% |
| even / over448 | -2.39% | 10/12 | - |
| every3 / power1 | -2.34% | 10/12 | +1.02% |
| every3 / eig | -2.33% | 10/12 | +1.44% |
| every3 / sketch | -2.19% | 10/12 | +0.40% |

The exact eigen-frame is no better than the one-pass sketch: at every join it is neutral (-0.10%), and with every3 it gives
-1.62% / -2.33%. Truncation quality per join is not the loss; the number of truncations is. every3 with the sketch (-1.86% on the four
nets) matches the frame-free exact old tier (mean -1.6%), so fewer joins already recover what old-tier exactness is worth.

## Cost

flopscope bills eigh(n) at 9 n^3 (4.5 units at n = 1024, 1.9 at 768, 0.38 at 448) and QR(1024 x 384) at 0.25 units.

Frame work per schedule in dense units (`frame_cost.py`, r35 dump sizes). The bank's join work totals 37.4 units, of which 15.3 is
frame work:

| schedule | sketch | power1 | over448 / warm | eig |
|---|---|---|---|---|
| every3 | 11.2 | 22.5 | 27.9 | 21.1 |
| every3b | 11.1 | 22.2 | 27.6 | 20.4 |
| even | 12.9 | 25.9 | 33.0 | 28.7 |

So a better frame costs +9 to +16 dense units per schedule, about +3-5% of the bill after Strassen-Winograd pricing.

Native schedules (lead's fable_join_v1 builds, r35 / r100):

| schedule | util | MSE |
|---|---|---|
| every3 | -0.76% / -0.9% | +0.3% / -4.1% |
| every3b | +2.1% / +3.3% | +2.3% / -3.7% |
| even | -0.8% / -0.4% | +2.4% / -2.5% |

## Adjusted estimate (raw x util)

| combination | estimate |
|---|---|
| **every3 / sketch** | about **-2.7% to -3.0%** (raw -1.9% to -2.2%, util -0.8%) |
| every3b / sketch | about -0.3% to +1.8% (raw -1.4% to -2.9%, util +2.1% to +3.3%) |
| every3 / power1 | about -0.1% to +0.2% (+0.15 points raw over the sketch for about +3% util) |
| even / over448 | about +2% |

Recommendation for candidate B: join schedule every3 (cohorts join at L7/10/13) with the bank's one-pass sketch. No frame upgrade.
