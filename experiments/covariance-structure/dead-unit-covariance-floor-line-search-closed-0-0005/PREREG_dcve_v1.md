# [opus-dcve] DCV-EARLY pre-registration (written 2026-10-16T16:07:20Z, before any DCV-EARLY data)

Lead GO 2026-10-16 ~16:07Z. Value-only cut: the dead-unit covariance roster floor (_DCV_FLOOR, read by _dcv_roster) goes deeper at
the EARLY layers only; layers 10-14 (and the L15 diagonal) keep the clean stack's DCV2 floor 0.002.

## Why it can pay (prior)
- DCV3 (0.003 at every layer) on 24 seeds was a wash: bill -0.262%, xMSE +0.225% (rate ~1.16), xnet -0.037%, aggregate +0.048%.
- Error-origin census (10-16): 52.5% of final-layer error is born at L13-15, i.e. ~17.5% per late layer vs ~4% per early layer:
  an early-born error weighs ~4x less per layer. DCV's bill cut is roughly per-layer uniform where the cut bites.
- If DCV3's accuracy loss follows that profile, L1-9 keeps ~60% of the bill cut (~ -0.16%) for ~25-35% of the loss (~ +0.06-0.08%):
  rate ~2-3, net ~ -0.1%.
- Caveat known before running: at layers 1-4 every unit is usually above the floor (live 1,024), and the roster rounds up to 64,
  so the deeper floor may bite only at L5-9; the seed-1 price decides.

## Arms (each = S2tr 46e9315d + _DCV_FLOOR = 0.002 + a per-layer floor table read by a _dcv_roster wrapper; layer = max(_R320_RESP) + 1)
- DCVE3: 0.003 at L1-9, 0.002 at L10-15.
- DCVE4: 0.004 at L1-9, 0.002 at L10-15.
Reference arm: DCV2 = S2tr + _DCV_FLOOR 0.002 (hunter's 3e2d0e73, its S8 runs in lead/runs/opus_ladder2r_v1/DCV2): every arm is judged
PAIRED vs DCV2 per seed (arm minus DCV2, both vs ctl_s2tr, noise-free cross-half U on 7103248-55).

## Gates (pre-registered)
1. Price on seed 1 (7103248): KILL if the bill saving vs DCV2 < 0.12%.
2. S8 (7103248-55): KILL if xMSE vs DCV2 > half the bill saving; PASS to A80 if xnet vs DCV2 <= -0.15% and aggregate < 0.
3. A80 confirm (7103080-095 vs ctl_s2tr_a80, paired vs the hunter's DCV2 A80 runs if present, else the DCV2 arm run fresh):
   24-seed xnet vs DCV2 <= -0.10% and A80 xnet <= 0 -> hand to release (value-only fragment) for the S3 line.
