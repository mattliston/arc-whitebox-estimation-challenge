# Total read-tail rank-8 completion at layers 7-14: 0.79% bill saving but 10.4x the incumbent tail error, closed

2026-10-16T23:32:46Z · experiment · rejected · measured

A saved-data test of completing the distinct old-plus-young column tail of the plane read (cutoff 1.0, retained-prefix power basis, rank 8) at layers 7-14 priced an optimistic 1.92756 GF (0.788056%) warm bill saving against the estimator submitted as #336412 (S2tr) tariff, but the fixed total-prefix rank-8 approximation of the newly covered band carried 10.447826x the incumbent entire-tail symmetric covariance error (7.35x even with an optimistic masked fit), so it was closed without a native run.

Recorded measurement:

    Optimistic saving 1.92756 G / 0.788056% bill; new band error 10.447826x incumbent (weighted directed 10.444983x); masked rank-8 fit 7.352470x (7.352166x).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_crunch_total_readtail_v1/audit_total_tail_v1.py](code/lead/runs/astra_crunch_total_readtail_v1/audit_total_tail_v1.py)
- [code/lead/runs/astra_crunch_total_readtail_v1/price_total_tail_v1.py](code/lead/runs/astra_crunch_total_readtail_v1/price_total_tail_v1.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py](code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4528](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4528)

<details>
<summary>All result files and records (4)</summary>

- [TARIFF_BOUND_v1.json](TARIFF_BOUND_v1.json) · result
- [SAVED_ERROR_v1.json](SAVED_ERROR_v1.json) · result
- [CLOSURE_v1.json](CLOSURE_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-total-read-tail-rank-8-completion-layers-7-14) · [Copy provenance](../../../PROVENANCE.json)

