# Grouped join and old-bank kernels: warm bill -0.347% but worst wall 121.23 seconds and risk 57.09%

2026-10-16T08:11:43Z · experiment · rejected · measured

The actual native and boundary implementation saved its predicted arithmetic but incurred far more requests and wall time than virtual pricing suggested. The combined source was retained only as a rejected record.

Recorded measurement:

    warm-.371/-.347/-.287/-.423/-.309%,mean-.347%;warmops+1670..2496;first+2130..3154;meanwall+10.9s,worst121.23s;first+4050/warm+2718requests;risk57.09%;residual.411s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_small_v1/build_smallsm_v1.py](code/lead/runs/opus_small_v1/build_smallsm_v1.py)
- [code/lead/runs/opus_small_v1/build_smallsm_v2.py](code/lead/runs/opus_small_v1/build_smallsm_v2.py)
- [code/lead/runs/opus_small_v1/price_small_v1.py](code/lead/runs/opus_small_v1/price_small_v1.py)
- [code/lead/runs/opus_small_v1/wallprice_v1.py](code/lead/runs/opus_small_v1/wallprice_v1.py)
- [code/lead/runs/opus_small_v1/bench_summary_v1.py](code/lead/runs/opus_small_v1/bench_summary_v1.py)
- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py](code/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py)
- [code/lead/runs/opus_governor_v1/candidate_GOV_v4g.py](code/lead/runs/opus_governor_v1/candidate_GOV_v4g.py)
- [code/lead/runs/opus_governor_v1/test_parity_v4g.py](code/lead/runs/opus_governor_v1/test_parity_v4g.py)
- [code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py](code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py)
- [code/release/rusage.py](code/release/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4027](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4027)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4010](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4010)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4012](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4012)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4021](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4021)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4034](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4034)

<details>
<summary>All result files and records (9)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_smallsm_template_v1.py.txt](fragment_smallsm_template_v1.py.txt) · result
- [fragment_smallsm_template_v1b.py.txt](fragment_smallsm_template_v1b.py.txt) · result
- [fragment_smallsm_v2.py.txt](fragment_smallsm_v2.py.txt) · result
- [PRICE_small_v1.json](PRICE_small_v1.json) · result
- [WALLPRICE_small_v1.json](WALLPRICE_small_v1.json) · result
- [BENCH_small_shard0.json](BENCH_small_shard0.json) · result
- [BENCH_small_shard1.json](BENCH_small_shard1.json) · result
- [fragment_pushsm_v2.py.txt](fragment_pushsm_v2.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-grouped-join-old-bank-kernels-warm-bill-0-347) · [Copy provenance](../../../PROVENANCE.json)

