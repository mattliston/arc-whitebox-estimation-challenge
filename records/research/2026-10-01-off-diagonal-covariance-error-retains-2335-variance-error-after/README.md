# Off-diagonal covariance error retains 23–35% of variance error after 64 eigen-directions

2026-10-01T19:20Z · finding · informational · measured

On network 35 the error of the carried covariance was decomposed. It is 99.6% off-diagonal (relative size 0.49 / 0.72 / 0.92% at layers 6 / 10 / 13); one direction near the mean holds 27-37% of it, but the next layer's per-unit variance error keeps 51-58% after removing 16 eigen-directions and 23-35% after 64, so the error is not low-rank.

Recorded measurement:

    off-diagonal share 99.6%; relative 0.49 / 0.72 / 0.92% at layers 6 / 10 / 13; one mean-near direction 27-37%; variance error remaining after 16 / 64 eigen-directions 51-58% / 23-35%

## Programs

- [a4_cov_anatomy.py](a4_cov_anatomy.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L12](../../TEAM_LOG_2026-10-01_to_10-22.md#L12)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-off-diagonal-covariance-error-retains-2335-variance-error-after) · [Copy provenance](../../../PROVENANCE.json)

