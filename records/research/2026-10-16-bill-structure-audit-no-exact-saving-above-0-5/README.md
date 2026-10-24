# Bill-structure audit: no exact saving above 0.5%; deeper products require 6,668 additional requests

2026-10-16T22:30:22Z · finding · rejected · measured

A teammate audit of the remaining bill-structure ideas on the estimator submitted as #336412 (S2tr) found the alternate basis already covers every tree path including the Smirnov batch kernel, a Smirnov depth+1 is infeasible (all 10 classes odd-32 at the next depth), the whole depth frontier is -0.42% on paper in about 100 moves of at most 0.06% each but costs +6,668 requests (net +0.98%), duplicate forward transforms are about 0.1%, float64 0.24%, view and binding 2.3% gross (0.5-0.7% realistic at more CPU) and QR to Cholesky-QR 0.2%. The live routing rules already price CPU and requests at zero. Nothing was built; no priced path to 1.46e-9 existed from the bill side.

Recorded measurement:

    Depth frontier: 55 moves -0.42% bill but +6,668 requests -> +0.98% net; 15 net-positive moves -0.073% for +520 requests and +1.9 s CPU. Duplicate forwards ~0.1% (11 repeats of 308); f64 0.24%; view/binding ~2.3% gross; QR -> CholQR 0.2%. NT join = wash (-1.3% MSE / +2.0% bill); pushSM on clean -0.08%.

## Programs

- [probe_treuse_v1.py](probe_treuse_v1.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_S2tr_v1.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_S2tr_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4519](../../TEAM_LOG_2026-10-01_to_10-22.md#L4519)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4521](../../TEAM_LOG_2026-10-01_to_10-22.md#L4521)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4522](../../TEAM_LOG_2026-10-01_to_10-22.md#L4522)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · result
- [depth_moves_v1.json](depth_moves_v1.json) · result
- [treuse_s2tr_7103256.json](treuse_s2tr_7103256.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-bill-structure-audit-no-exact-saving-above-0-5) · [Copy provenance](../../../PROVENANCE.json)

