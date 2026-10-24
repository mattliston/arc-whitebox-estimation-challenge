# Data-movement census: 1.62% of warm bill; float32 covariance state rejected

2026-10-16T14:19:45Z · finding · rejected · measured

A per-site census of the pubmaxapg2 parity copy (captured setup, network 100) priced data movement at 1.62% of the warm bill (2.33% on the first network, where lazy binds add 0.40%), with the largest sites the Smirnov linear-group stack, the unique-rows scatter, lazy input-view binds and the quadrant reshape of each tree output. A dtype-aware second pass put float64 work at 0.507% of the warm bill spread over about 40 sites, so a float32 covariance state or an f64 to f32 sweep is worth at most about 0.25% and touches precision-critical means; it was killed. The remaining exact harvest was the Smirnov linear-group stack (0.13%) and a prebind refit (lazy view binds 0.087% warm).

Recorded measurement:

    Data movement 1.62% of warm bill (net 100), 2.33% on the first net; top warm sites: Smirnov linear-group stack 0.130%, unique-rows scatter 0.100%, lazy _ipt_view binds 0.087%, _l3o_last quadrant reshape 0.078%, row gathers/scatters 0.08%, _ps32 f64 to f32 conversions 0.052%, kernel operand loads ~0.33%; plausibly exact-rewritable ~0.3%. Float64 = 0.507% of the warm bill (largest: _ps32 conversions 0.051%, f64 matvecs 0.024-0.027% each, pair-batch astype/outer/sum 0.012% each).

## Programs

- [site_census_v2.py](site_census_v2.py)
- [../../../instruments/release-pipeline-combines-boundary-gates-captured-setup-parity-prefire/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py](../../../instruments/release-pipeline-combines-boundary-gates-captured-setup-parity-prefire/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4206](../../TEAM_LOG_2026-10-01_to_10-22.md#L4206)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4213](../../TEAM_LOG_2026-10-01_to_10-22.md#L4213)

<details>
<summary>All result files and records (3)</summary>

- [census_pg2.log](census_pg2.log) · log
- [census_pg2_dtype.log](census_pg2_dtype.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-data-movement-census-1-62-warm-bill-float32-covariance) · [Copy provenance](../../../PROVENANCE.json)

