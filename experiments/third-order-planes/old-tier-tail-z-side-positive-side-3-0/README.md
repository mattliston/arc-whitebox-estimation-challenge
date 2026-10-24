# Old-tier tail z by side: positive side 3.0 buys -0.50% MSE for +0.51% bill; negative side and deeper cuts lose

2026-10-16T14:32:34Z · experiment · rejected · measured

The old-tier tail threshold (OTT: rows of the old plane tier beyond a pre-activation ratio z are reconstructed rather than kept) was split by side on the estimator submitted as #336412 (S2tr). Raising only the positive side from 2.5 to 3.0 buys accuracy at about 1:1; raising the negative-dead side buys nothing, so the accuracy lives in the strongly active rows. Cutting the negative side deeper (2.25) or both sides (2.25) lost; the deeper direction pays nowhere in this share.

Recorded measurement:

    OTTP30 (positive 2.5 -> 3.0; df2db670): xMSE -0.496% (SE .34, 4/8), bill +0.505%, xnet +0.009%. OTTN30 (negative side 3.0): xMSE +0.082% (SE .32), bill +0.535%, xnet +0.617%. OTTN225 (_OTT_NEG_Z 2.25; 44650ba8): xMSE +0.639% (SE .37, 3/8), bill -0.097%, xnet +0.543%. OTTN20 / OTT225 stopped mid-arm.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_ladder2h_v1/build_ladder2h_v3.py](code/opus_ladder2h_v1/build_ladder2h_v3.py)
- [code/opus_ladder2h_v1/build_layerr_v1.py](code/opus_ladder2h_v1/build_layerr_v1.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4208](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4208)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4232](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4232)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4237](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4237)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Old-tier tail reconstruction on selected frame coordinates failed the arithmetic cost test](../old-tier-tail-reconstruction-selected-frame-coordinates-failed-arithmetic/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-old-tier-tail-z-side-positive-side-3-0) · [Copy provenance](../../../PROVENANCE.json)

