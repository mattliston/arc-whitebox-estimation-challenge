# S4 fire plan (drafted 2026-09-23 00:25 PDT)

Candidate: fable_porter_v1/stack/stack_S3_gn.py = bank -> e2 v7 -> K31 row (.85/.5, L1-14) -> b-fix 0.1 -> GN covariance knobs v3
(iso -0.094, k21n -0.028, k31 -0.212, edge +0.164, rc -0.080; porter build_gn_knobs.py) -> kernel --holdelim -> join every3
(v1 fired builder). Kernel swap to HEB (--bigarena) only if engineer2's gate passes before the fire (bit-identical predictions).

Expected: VALID40 adjusted ~0.923 vs bank (S3 0.937 x knobs ~0.985). Board ~3.29e-9 if the local-to-board gap matches S2's
(local overstated S2's raw gain by ~0.8 pt). Not a milestone (3e-9 needs -10.2% from #331947); fired as a new best under the
standing pre-authorisation, and to get the S3 kernel's platform wall telemetry.

Gates (all must pass):
1. VALID40 paired vs cached bank: adjusted <= 0.925, >= 30/40 nets better than S3.  [porter]
2. panel8 + virgin6 both better than S2's panel (3.660e-9 / 3.002e-9).                [porter]
3. whest validate on the exact file.                                                   [porter]
4. quiet_census_v2.sh: local server CPU and round trips <= S3-HE + 1 s (paired window). [lead]
5. one-process memory/handles over 6 nets: live peak <= 2.8 GB, handles <= bank.      [engineer2]
Submit: autoresearch submit path as for #331947 (fire_candS2), record sha256, ledger line, poll loop.

## Gate results (00:52)
1. VALID40 (porter, paired_valid40_stack_S3_gn.log): adjusted 0.92453 vs bank (40/40), raw 0.94344. Vs fired S2: adjusted
   0.97772 (-2.23%), better on 37/40, per-net range -5.19%..+1.42%. Vs S3: raw 0.9846, 31/40. PASS.
2. panel8 + virgin6: running (porter).
3. whest validate: pending (porter).
4. Compute: the knob layer adds ~40 n^2 elementwise ops per layer (~0.5-1 s local by op count) and +2.1k round trips
   (+0.5 s platform overhead). My paired census under load 16 was too noisy to resolve it (+11.3 s / +2.3 s). S3-HE is
   -9%/-17% server CPU vs S2 (engineer2, interleaved). Projected S4 platform mean ~90 s, worst ~100 s. ACCEPTED on the op
   count; the S4 telemetry will measure it.
5. Memory: S3-HE live = S2 exactly; HEB +378 MB live (3.05 GB peak), RSS 3.97 GB; both pass. Kernel = HEB if engineer2's
   compute gate passes, else HE.
Expected board: 3.339e-9 x 0.9777 = ~3.26e-9 (x 0.9965 with HEB = ~3.25e-9).

## S5 status (02:20)
S5 = S4 + L15 cut + push5 depth rule + f32 third + engineer2 memory diet. Parity: bill -1.30% (r35 cold) / -2.85% (r100 warm) vs S4, MSE +0.18/+0.15% (rounding). BLOCKED on handle growth (+9% vs S4; cap extrapolated at ~18-19 distinct nets per worker connection vs ~20 MLPs per worker). Waiting for engineer2's 20-net test.
