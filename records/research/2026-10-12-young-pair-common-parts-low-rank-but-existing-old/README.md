# Young-pair common parts are low rank, but the existing old-tier basis captures only 87-96%

2026-10-12T03:13:04Z · finding · informational · measured

A read-only probe on S48b (network 7100136, local numpy) measured the lam^2-weighted energy of the young pairs' common parts c = (p+ + p-)/2 kept at layers 8-14 by (i) the old tier's basis pushed to that layer and (ii) the best rank-r basis of the c's themselves: the old basis keeps 0.874-0.960 while the best rank-384 basis keeps 0.99944-0.99980 (rank 512: 0.99982-0.99995), 100-300x less loss. A pair-split read on the old tier's basis was therefore out, and any right-factor-only form needed its own young-c basis.

Recorded measurement:

    L8 (4416 pairs): old 0.9601 | best 256 / 320 / 384 / 512 = 0.99798 / 0.99899 / 0.99944 / 0.99982; L9: 0.9601 | 0.99837 / 0.99917 / 0.99954 / 0.99986; L10: 0.9216 | 0.99863 / 0.99929 / 0.99961 / 0.99988; L11: 0.9131 | ... 0.99969 / 0.99991; L12: 0.9110 | ... 0.99974 / 0.99993; L13: 0.8740 | ... 0.99977 / 0.99994; L14: 0.8836 | ... 0.99980 / 0.99995; the old basis keeps 0.68-0.81 of the differences h; h is 1.0-2.1% of the pairs' lam^2 energy

## Programs

- [probe_ycbasis_v1.py](probe_ycbasis_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1283](../../TEAM_LOG_2026-10-01_to_10-22.md#L1283)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1284](../../TEAM_LOG_2026-10-01_to_10-22.md#L1284)

<details>
<summary>All result files and records (2)</summary>

- [probe_ycbasis_s48b_136.json](probe_ycbasis_s48b_136.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-young-pair-common-parts-low-rank-but-existing-old) · [Copy provenance](../../../PROVENANCE.json)

