# Per-network choice among existing variants has oracle gain at most 0.19%

2026-09-25T08:18Z · experiment · rejected · measured

The best per-network selection barely beats the best single candidate and is comparable to selection from numerical noise. A predictive router was not pursued for those variants.

Recorded measurement:

    - 2026-09-25 02:00 analyst: PER-NET ADAPTIVE ALLOCATION, STEP 1 STOPS (fable_analyst_v1/RESULT.md section 38; oracle_router.py on the porter's native VALID40 rows, adjusted = MSE x util per net). Per-net oracle vs the best single variant: S21 vs S21jn -0.07% (5/40 flips), S21s1 vs s1=0 0.00% (s1=0 wins 40/40), S21s1 vs S22 -0.11%, s1=0 vs S22 -0.16%, S23 vs S23+Nystrom star -0.08% (10/40), 5-way S21 family -0.19%, S22/S23/Nystrom -0.08%. All under the 0.5% bar and at the rounding noise floor (~0.12% for equal cards). Routing value could only live in cost dials with net-varying raw price: the S2-era old-tier rank 320 at L10/L13 (4 nets, torch) had a -0.6..-0.7% oracle; testing that on today's card needs one cost dial's 40-net rows plus a deterministic sketch-energy rank rule. Step 2 not run.
    - 2026-09-25 01:18 lead: PER-NET ROUTING over existing variants CLOSED (analyst §38): per-net oracle <= 0.19% over the best single variant, at the rounding-noise floor (equal cards would show ~0.12%). Successive physics pairs win almost everywhere. The one routable dial: the old-tier rank (S2-era torch per-net oracle 0.6-0.7%). Analyst next: ENERGY-ADAPTIVE join rank (per net and per join, r in {256, 320, 384} by a captured-energy threshold tau) vs fixed rank at MATCHED average bill, torch VALID16; bar >= 0.3% raw at equal estimated bill.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_analyst_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/oracle_router.py](code/oracle_router.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L859](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L859)
- [TEAM_LOG_2026-09-22_to_09-26.md#L860](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L860)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT-1a40d4c3.md](../../third-order-planes/gain-mixture-correction-half-strength-native-adjusted-ratio-1/RESULT-1a40d4c3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-per-network-choice-among-existing-variants-has-oracle-gain) · [Copy provenance](../../../PROVENANCE.json)

