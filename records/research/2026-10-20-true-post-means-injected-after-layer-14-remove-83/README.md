# True post means injected after layer 14 remove 83% of final MSE; true post-variance diagonals change it only +-1%

2026-10-20T02:33:00Z · finding · informational · measured

Two in-process oracles on fresh networks with 64M-sample truths: injecting the true post-activation means after layer l removes 83% of the final MSE at l = 14 (63% at 13, 56% at 12, 35% at 10, 20% at 8), so the final error is inherited mean error; injecting the true post-activation variance diagonal at any layer or range changes the final MSE by +-1%, so the covariance diagonal is right and each layer's own contribution enters through the off-diagonal (plane) content via the variance and the k3 path.

Recorded measurement:

    post-mean oracle (seed 7100005): L14 -83%, L13 -63%, L12 -56%, L10 -35%, L8 -20%; post-variance diagonal oracle (L14/13/12/10/8, L12-14, L8-14, L1-14; 3 fresh nets): +-1%

## Programs

- [probe_postmean_oracle_v1.py](probe_postmean_oracle_v1.py)
- [probe_postvar_oracle_v1.py](probe_postvar_oracle_v1.py)
- [candidate_pvdir_sci_v1.py](candidate_pvdir_sci_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6107](../../TEAM_LOG_2026-10-01_to_10-22.md#L6107)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6113](../../TEAM_LOG_2026-10-01_to_10-22.md#L6113)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6117](../../TEAM_LOG_2026-10-01_to_10-22.md#L6117)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-true-post-means-injected-after-layer-14-remove-83) · [Copy provenance](../../../PROVENANCE.json)

