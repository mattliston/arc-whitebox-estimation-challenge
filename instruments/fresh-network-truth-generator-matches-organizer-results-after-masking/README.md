# Fresh-network truth generator matches organizer results after masking a GPU random-number defect

2026-10-01T22:58Z · tool · informational · measured

A GPU Monte Carlo truth generator for freshly drawn networks (iid N(0, 2/1024) float32 weights, seeds 7,100,000 + k) was certified on public network 35 against the organizers' truth: the mean-squared difference along the covariance principal axes equals the predicted sampling noise to 0.964-1.040 over 16 layers. torch 2.7 randn on MPS returns exactly -14.372 once per about 8.4 million values; the generator masks those values and flips the sign of every second batch. It produced 155-180 networks per day as an untouched check set.

Recorded measurement:

    ratio to predicted sampling noise 0.964-1.040 (mean 0.996, chance sd 4.4%); mean direction within 1.3 se at every layer; final-layer truth noise 1.05e-9 (public 35), 1.33e-9 (seed 7100000); RNG defect: 490 in 4.0e9 values = -14.372, input variance +2.5e-5, skew -3.9e-4, excess kurtosis +0.005; after masking 1e11 values: mean within 4e-6, variance within 2e-6, skew within 4e-6, excess kurtosis -1e-4; 480-555 s per network (134k samples/s quiet, 115k loaded)

## Programs

- [code/cert_eval.py](code/cert_eval.py)
- [code/mc_truth.py](code/mc_truth.py)
- [code/prod.sh](code/prod.sh)
- [code/loader.py](code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L53](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L53)
- [TEAM_LOG_2026-10-01_to_10-22.md#L54](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L54)
- [TEAM_LOG_2026-10-01_to_10-22.md#L55](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L55)
- [TEAM_LOG_2026-10-01_to_10-22.md#L58](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L58)

<details>
<summary>All result files and records (2)</summary>

- [cert_rank0035.md](cert_rank0035.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-01-fresh-network-truth-generator-matches-organizer-results-after-masking) · [Copy provenance](../../PROVENANCE.json)

