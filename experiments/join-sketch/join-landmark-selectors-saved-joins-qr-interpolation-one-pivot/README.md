# Join landmark selectors on saved joins (QR interpolation, one-pivot, core diagonal, pair rotation): no consistent gain

2026-10-17T09:43:26Z · experiment · rejected · measured

Five saved-data selector or completion variants of the Nystrom join were closed without a native run: interpolating the leading 256 coefficients by rows (hybrid/orthogonal residual energy ratio 2.14-2.65); a one-pivot landmark selector (mean -0.14% of compression residual energy, 2/6 wins); replacing the stored core diagonal (6.1% local operator error at layer 10, like the earlier layer-7 null); a read-only fixed-64 coordinate selector (0.43% weighted symmetric error recovery, 4 of 64 swaps per block); and a read-only pair rotation of the tail 64 rows (4.26% weighted covariance error recovery against a uniform cut).

Recorded measurement:

    Hybrid interpolation: residual-energy ratios hybrid/orth 2.141-2.645 (mean 2.3668), full interp 3.2690; exact trailing 128 removes 39.95% of damage. One-pivot: join energy changes -1.133%, +.205%, +.076%, +.046%, -1.232%, +1.169% (mean -.1449%). Core diagonal: 6.12-6.16% local error^2 at L10 (L7 fixed8 null raw -.000265%, bill +.002900%). Coordinate energy selector: .431766% weighted recovery, raw off-diagonal -.924199%, +7.054 MF. Pair rotation: factor energy recovered 3.79/4.32%, weighted symmetric covariance error 4.2649%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_crunch_hybrid_interp_v1/saved_v1.py](code/astra_crunch_hybrid_interp_v1/saved_v1.py)
- [code/astra_crunch_onepivot_v1/saved_v1.py](code/astra_crunch_onepivot_v1/saved_v1.py)
- [code/astra_crunch_physics_v1/core_diagonal_v1/saved_v1.py](code/astra_crunch_physics_v1/core_diagonal_v1/saved_v1.py)
- [code/astra_crunch_physics_v1/read_coordinate_energy_v1/saved_v1.py](code/astra_crunch_physics_v1/read_coordinate_energy_v1/saved_v1.py)
- [code/astra_crunch_physics_v1/read_pair_rotation_v1/saved_oracle_v1.py](code/astra_crunch_physics_v1/read_pair_rotation_v1/saved_oracle_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4791](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4791)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4793](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4793)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4865](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4865)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4867](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4867)

<details>
<summary>All result files and records (9)</summary>

- [CLOSURE_v1.json](CLOSURE_v1.json) · record
- [CLOSURE_v1-5ea22e7c.json](CLOSURE_v1-5ea22e7c.json) · record
- [CLOSURE_v1-efcf5333.json](CLOSURE_v1-efcf5333.json) · record
- [CLOSURE_v1-90c6ff84.json](CLOSURE_v1-90c6ff84.json) · record
- [CLOSURE_v1-3463eadc.json](CLOSURE_v1-3463eadc.json) · record
- [RESULT_v1.json](RESULT_v1.json) · result
- [GUARD_v2.json](GUARD_v2.json) · result
- [PILOT_DECISION_astra_v1.json](../exact-diagonal-first-rank-384-join-core-pilot-adjusted/PILOT_DECISION_astra_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-join-landmark-selectors-saved-joins-qr-interpolation-one-pivot) · [Copy provenance](../../../PROVENANCE.json)

