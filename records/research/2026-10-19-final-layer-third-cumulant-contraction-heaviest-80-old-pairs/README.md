# Final-layer third-cumulant contraction: heaviest 80% of old pairs leave 2.5-3.0% relative error

2026-10-19T10:38:18Z · finding · informational · measured

Probes of the final-layer k3 contraction on net 7103256 (672 rows, 16,640 old and 7,872 young columns): the cubic contraction concentrates on the old side (keep 80% of old columns -> k3 error 2.5%, 70% -> 4.1%, 50% -> 9.4%) but not on the young side (keep 90% -> 10%). The roster comes in pairs whose pushed columns correlate +0.96 with lam_t = -lam_s, so a pair's k3 contribution is a 17% residual of its gross and any pair cut must drop both legs; a pre-push proxy ranks old pairs at spearman 0.84 and the net pre-push cube on the previous rows ranks inherited young pairs at 0.991.

Recorded measurement:

    old columns keep 80% -> 2.5%, 70% -> 4.1%, 50% -> 9.4% (proxy spearman 0.992); young keep 90% -> 10%; 12,256 pairs; sum|net| / sum(|t|+|s|) = 0.17; pair oracle old 80% -> 3.0%, 70% -> 5.3%, 60% -> 7.9%, 50% -> 11.5%; young 90% -> 3.1%, 80% -> 7.8%; old proxy |lam_t||f_t|^3 + |lam_s||f_s|^3 spearman 0.84 (80% -> 3.2%, 70% -> 5.5%); 3,936 young pairs = 3,392 inherited + 544 newborn, pre-push cube spearman 0.991 (90% -> 2.6%, 80% -> 6.5%), |lam||p|^3 proxies 0.69

## Programs

- [probe_l15k3_v1.py](probe_l15k3_v1.py)
- [probe_l15k3_v2.py](probe_l15k3_v2.py)
- [probe_l15k3_v3.py](probe_l15k3_v3.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5931](../../TEAM_LOG_2026-10-01_to_10-22.md#L5931)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5933](../../TEAM_LOG_2026-10-01_to_10-22.md#L5933)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5934](../../TEAM_LOG_2026-10-01_to_10-22.md#L5934)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5936](../../TEAM_LOG_2026-10-01_to_10-22.md#L5936)

<details>
<summary>All result files and records (4)</summary>

- [l15k3_relstk7_7103256.json](l15k3_relstk7_7103256.json) · result
- [l15k3v2_relstk7_7103256.json](l15k3v2_relstk7_7103256.json) · result
- [l15k3v3_relstk7_7103256.json](l15k3v3_relstk7_7103256.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-final-layer-third-cumulant-contraction-heaviest-80-old-pairs) · [Copy provenance](../../../PROVENANCE.json)

