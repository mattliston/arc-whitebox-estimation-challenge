# FASTER PATH MEMO — perfect mean+variance supply (2026-09-01, fork)

Written by the strategy fork while the main session hill-climbs the
tracker. Every number below was measured in this pass with a
beta-parametrized copy of the consumption instrument
(`public_g19_attribution_v1/consumption_betas.py`; the original was
not modified). Frame: board raw 2.94e-8; instrument baseline
(deployed card, rank35) 2.676e-8; goal <1e-8.

## The two findings that reorder everything

**F1. The mean channel converts automatically — the goal is reachable
inside the existing consumption, but only with ALL THREE channels.**
`SUPPLY=oracle` (truth v, k3, k4 through the instrument's linear
mean-integral): rank35 corrected **7.09e-9 (−73.5%)**, rank100
**9.89e-9 (−64%)** — both UNDER the goal, matching the ladder's exact
m+v (7.48e-9). The earlier "v-oracle = 1.25–1.46e-8" was the v-arm
ALONE (only_v 47.4% / 56.4%). At truth, k3 is worth +10.0 and k4 +7.8
on rank35 and the three arms interact to 73.5. Assumption (d) is
resolved: m needs no separate supply object — the instrument's
r-recursion (`r = phi*(r@W) + dds`) IS the exact linear m-integral —
but v-alone caps at ~1.4e-8. **The k3/k4 channels are not a 2–8%
sideshow; they are ~26 of the 73 points.**

**F2. The integration betas are stale by ~2x — the single largest
cheap lever on the table, deploy-grade, frozen-stable.**
The fixed universal betas (v 0.82, k3 0.33, k4 0.16) were LOO-fit for
an older supply. On the DEPLOY-FRAME supply the card can reproduce
(`dvhat_*_dep.npz`), optimal β_v is **0.40 (r35) / 0.435 (r100)**:

| supply / net | only_v at β=0.82 | only_v at optimal β | joint 3-arm at opt |
|---|---|---|---|
| _dep rank35 | −1.1% | **+10.6%** (β 0.40) | +14.5% |
| _dep rank100 | +3.4% | **+15.4%** (β 0.435) | +18.3% |
| _dep rank100 at r35's β=0.40 (FROZEN) | — | **+15.3%** | — |
| tracker-stacked rank35 | +2.2% | +10.9% | +13.6% |

The tracker's whole contribution at optimal betas is +0.3–0.6 on top
of the properly-scaled chain — consistent with its joint corr 0.811 vs
0.79 (Δcorr² 3.4 pts of the 47-pt v-pool ≈ +1%). The chain's own
supply, correctly scaled, is worth 10–15 points that the deployed
integration is leaving on the table (or under/over-applying —
VERIFY, see Rank 1).

## Ranked moves (honest corr²-per-hour)

**Rank 1 — Beta re-integration of the existing deploy-frame supply.**
Mechanism: refit (LOO across the panel) the three consumption betas
for the supplies the card actually carries; expected only_v +10–15%,
joint +14–18% on both nets (measured in-sample; β_v frozen-stable
0.40↔0.435). Fence check: never run with current supplies — the
0.82 dates from PERFECT_STATE_CEILING on an earlier carrier. Cost:
hours (the LOO needs the per-net r-vectors: DUMP_R=1 across the 8
panel nets — minutes each). Value/hour: by far the highest. THE
VERIFICATION IT NEEDS: confirm what the deployed card #328622
actually applies (which dvhat frame, which betas). If the card
applies this same supply at 0.82, the fix is a one-scalar change to
~1.2x total (0.82+0.40) — confirm by regenerating the deployed
output with the retuned beta and re-grading. If the card's internal
supply differs, `_dep` is additive at 0.40. Either way it ships
without any new mathematics. Decisive test: 30 minutes.

**Rank 2 — Supply k3 and k4 at truth-class, not just v.**
Mechanism: the oracle says v-alone = 47%, all-three = 73% (goal);
the k3/k4 arms at truth are +10/+8 with strong positive interaction.
Current k3/k4 supplies (kt-carrier) are NEGATIVE on rank100 (−8.1/
−7.5 at fixed betas; at optimal betas +2.2/+0.6 on r35). The tracker
produces k3/K21-planes for free (the same T-emission words; the
K21r assembly certified (1,1,1,½,1)) — grade its k3u supply through
the consumption's k3 arm. Expected: the k3-arm from 0 → +3–6 with
positive interaction on v. corr²/hour: moderate; the machinery
exists (run_g3_engine_delta's K21r side). Decisive test: emit k3u
per layer from the tracker state (diag of K21r), feed as k3hat, read
only_k3_at_opt. ~2 hours.

**Rank 3 — Grade candidates by the consumption's own value metric,
not per-layer pushed-v corr (assumption (a)).** The per-layer corr
ignores the L16 transport weighting (`phi*(r@W)` survival) and the
three-arm interaction. `consumption_betas.py` at optimal betas runs
in seconds — make only_v_at_opt / joint_at_opt the design compass.
This alone would have caught the beta staleness weeks ago. Zero cost;
immediate.

**Rank 4 — The deep layers are NOT value-dead; the "value hump" is a
fixed-beta artifact (question 3).** Solo slots at L5/L9/L12 each read
~10% at optimal beta (they all inherit the chain-rescale, so the
per-slot increment is small — but nothing cancels). The main session
should retire the hump law and re-run the layer-value map at optimal
betas; deep layers with the largest Δv targets are back in play for
the tracker's supply. Cost: minutes.

**Rank 5 — Exact resolvent T-Gram in the G3 channel (assumption
(c)).** The word shortcut is certified at BIRTH (R² 0.9989) but the
channel is evaluated on assimilated deep states; T-3's emission gates
at L2–L6 read 0.92–0.98 with the resolvent form vs the ladder's
failures — the word model on deep states is unmeasured. Expected:
G3 channel 0.111 → 0.15–0.2 class (corr² +0.01–0.03 of the v-pool
→ ~+1–2% only). Real but small; do it after Ranks 1–3. Cost: the
double-t-quadrature Gram (~half a day).

## Assumption audit

(a) Pushed-v correlation as the grade — REPLACE with the consumption's
optimal-beta value (Rank 3). Corr² is the right intuition but the
L16 transport weights layers unevenly and the arms interact.
(b) Card anchoring each layer — KEEP (the λ-dial curve is
single-peaked at 0.75; hard assimilation kills signal; no assimilation
drifts). A subspace-trusting hybrid was tested (spectral split) and
was a monotone null.
(c) Word-basis T at depth — unmeasured on assimilated deep states;
Rank 5.
(d) m-channel conversion — CONVERTS automatically through the
instrument's linear integral (oracle → 7.09e-9). No new m-object
needed. The goal requires all three channels at near-truth.
(e) LOO-beta integration — confirmed as the top multiplier, but the
beneficiary is the CHAIN's existing supply (+10–15%), not the
tracker (+0.3–0.6). It should have been run before any tracker work.

## Unexamined angles (question 3)

- Cum3-tensor path as deployable: the instrument-grade `tensor_planes`
  supply reads only marginally better than `_dep` at optimal beta
  (10.3 vs 10.6 on r35) — the deploy frame already captures nearly
  all of the tensor's value. No reason to pursue billing the tensor.
- Assimilation targets beyond C: k3-assimilation was a null (τ≈1);
  K21-plane assimilation untested but the tracker's supply increment
  is small at optimal betas regardless — low priority until Rank 2
  shows the tracker's k3 channel matters.
- Multi-net shared structure: β_v identical across nets (0.40/0.435)
  is itself the shared structure — exploit it as a frozen global.
- Deep layers: see Rank 4 — reopened.

## What "faster" means numerically

Board 2.94e-8 → with Rank 1 alone (frozen +10–15% only_v; joint
+14–18%): **~2.5–2.6e-8** (−12 to −17% raw), deployable this week,
co-fires with the f32 pack for ~−17 to −22% adjusted — well over the
meaningful bar. The goal (<1e-8) still needs the k3/k4 channels
lifted toward truth (Rank 2 is the entry) and v-supply corr into the
0.9s; the oracle proves the instrument converts all of it (7.1e-9)
once supplied. The tracker remains the only route to the 0.9s, but it
is not the fastest lever today.
