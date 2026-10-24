# Designated primary and fallback evaluated: predicted podium probability rises from 31.6% to 34.8%

2026-10-12T00:40:07Z · decision · informational · measured

The pair model (pair_mc_twin.py on candidate generation 48 (S48)'s own graded page, rivals' pairs from the current scan, P(loss) from loss_model_v3 first-priced at 7.35 us) compared designation pairs: (a) S48 + S47e P(podium) 31.6% at 100 networks / 30.2% at 300; (c) S48 + S48b (an identical-score fallback) 32.2% / 32.0%; (b) twin + S48 34.8% / 34.2%; (d) twin + S48b the same as (b). Split properly, the identical-score fallback ((a) -> (c)) is worth +0.6 points at 100 networks and +1.8 at 300, and the twin's own -0.094% of metered FLOPs ((c) -> (b)) +2.6 and +2.2, so close to a place line the twin's metered FLOPs pays; after grading the designated pair became twin #335451 primary + S48 #335411 safe, and the candidate generation 50 (S50) pair was later priced +2.3 points over it.

Recorded measurement:

    (a) 31.6% / 30.2%, expected place 4.03 / 4.11; (b) 34.8% / 34.2%, 3.96 / 4.05; (c) 32.2% / 32.0%; twin bill vs S48 -0.170% on network 35, -0.089% on network 100, about -0.094% over 6 first + 94 warm; P(loss) S48 3.0% / 8.8%, S47e 2.9% / 8.4%, twin 4.5% / 12.5% (with the S48b prebind); S50 twin + S50 safe 18.3% / 18.0% vs twin v2 + S48 16.0% / 15.7%

## Programs

- [../2026-10-11-measured-variant-recipe-drops-fused-trees-deeper-request-caching/pair_mc_twin.py](../2026-10-11-measured-variant-recipe-drops-fused-trees-deeper-request-caching/pair_mc_twin.py)
- [../2026-10-11-residual-time-model-has-heavy-tail-retain-free-request/loss_model_v3.py](../2026-10-11-residual-time-model-has-heavy-tail-retain-free-request/loss_model_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1248](../../TEAM_LOG_2026-10-01_to_10-22.md#L1248)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1252](../../TEAM_LOG_2026-10-01_to_10-22.md#L1252)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1268](../../TEAM_LOG_2026-10-01_to_10-22.md#L1268)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1335](../../TEAM_LOG_2026-10-01_to_10-22.md#L1335)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-designated-primary-fallback-evaluated-predicted-podium-probability-rises-31) · [Copy provenance](../../../PROVENANCE.json)

