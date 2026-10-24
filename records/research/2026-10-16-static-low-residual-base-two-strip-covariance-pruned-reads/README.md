# Static low-residual base with two-strip covariance and pruned reads: +1.466% bill, not submitted

2026-10-16T15:15:01Z · decision · rejected · measured

Three static low-residual builds were released and gated: LR2 (the estimator submitted as #336412 (S2tr) -> dm2 -> lazybind v2 -> two-strip -> SAFE prebind patch v2 -> five-full -> prune r005 -> DC35 -> QL/compute off) passed every count gate and became the base every gain was stacked on; LR1 (prune r002 instead of r005, ubown/cluster off, a static cold-first rule) also passed but was more expensive on the 8-network public mix; LR3 (LR2 with r002) was rejected as base because its 66 extra Smirnov plans cost +2.1-3.5% client Python instructions per request. LR2 projected 1.4940e-9 (+1.23% vs #336412) and was never fired: a gate-fitting build, not a board gain. The 8-network worker-like session, not the 2-network boundary pair, became the referee for public-mix bills.

Recorded measurement:

    LR2 9ef4b5e5: requests normal 41,911 first / 40,744 warm, slow 42,304 / 40,945; worst wall 92.1 s (mean 85.3); setup 4.442 / 4.585 s; residual projection median .317 / max .376; bill vs S2tr +1.561% first / +1.455% warm (public mix +1.466%); PREREG 1.4940e-9. LR1 ebeca524: normal 41,173/40,862, slow 41,583/41,063, worst 96.4 s, prereg 1.4959e-9; LR1 vs LR2 public mix +0.08% (cold +0.74%). LR3 83b719e5: slow 42,640/41,112, worst 100.3 s, bill -0.2% but instruction-weighted median .327 vs LR2 .317, P(any fail) 3.5% vs 0.4%.

## Programs

- [cand_lowres_LR2_v3.py](cand_lowres_LR2_v3.py)
- [cand_lowres_LR1_v3.py](cand_lowres_LR1_v3.py)
- [cand_lowres_LR1b_v3.py](cand_lowres_LR1b_v3.py)
- [compare8_v1.py](compare8_v1.py)
- [../../../instruments/single-attempt-submission-uploader-records-one-qualified-archive-receipt/code/fire_once.py](../../../instruments/single-attempt-submission-uploader-records-one-qualified-archive-receipt/code/fire_once.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4252](../../TEAM_LOG_2026-10-01_to_10-22.md#L4252)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4254](../../TEAM_LOG_2026-10-01_to_10-22.md#L4254)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4258](../../TEAM_LOG_2026-10-01_to_10-22.md#L4258)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4261](../../TEAM_LOG_2026-10-01_to_10-22.md#L4261)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4263](../../TEAM_LOG_2026-10-01_to_10-22.md#L4263)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4265](../../TEAM_LOG_2026-10-01_to_10-22.md#L4265)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4269](../../TEAM_LOG_2026-10-01_to_10-22.md#L4269)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4278](../../TEAM_LOG_2026-10-01_to_10-22.md#L4278)

<details>
<summary>All result files and records (6)</summary>

- [PREREG_lowreslr2_v1.json](PREREG_lowreslr2_v1.json) · record
- [PREREG_lowreslr1_v1.json](PREREG_lowreslr1_v1.json) · record
- [RELEASE_CONFIRMED_v1.json](RELEASE_CONFIRMED_v1.json) · record
- [COUNT_GATES_v4_rel_lowreslr2_20261016_opus_v1.json](COUNT_GATES_v4_rel_lowreslr2_20261016_opus_v1.json) · result
- [ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json](../2026-10-15-per-network-wall-limit-corrected-120-seconds-inherited-100/ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Static low-residual stack projects 1.4747e-9 on a boundary pair but only ties the baseline on eight networks](../../../experiments/submission-process/static-low-residual-stack-projects-1-4747e-9-boundary/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-static-low-residual-base-two-strip-covariance-pruned-reads) · [Copy provenance](../../../PROVENANCE.json)

