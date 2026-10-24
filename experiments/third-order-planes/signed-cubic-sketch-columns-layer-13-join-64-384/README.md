# Signed cubic sketch columns in the layer-13 join (64 of 384): plane error x1.10-1.13 for +0.17% bill, closed

2026-10-17T00:12:00Z · experiment · rejected · measured

Two constructions replaced 64 of the 384 static Gram columns of the layer-13 join with signed cubic contractions (centered Gaussian sketch J(lambda((J Omega)^2 - norm^2/m)); then exact contractions along the next weight columns W14[:, ::16]). On the saved rank-651 join both raised the pushed third-order plane error (K21 x1.1307 and x1.1025) and the held-out cubic contraction error, at a warm tariff of about +0.17% and 120 charged ops, so both were closed without a native run.

Recorded measurement:

    Cubic sketch64: K21 norm error x1.130737, K3 x1.122741, mean-sensitive K3 x1.104135, held-out signed tensor error^2 x1.323877; tariff +.169848% (800x4800) / +.167592% (832x4544), +120 charged ops. Next-weight cubic64: K21 x1.102512, k3 x1.058081, next-mean-weighted k3 x1.014183, held-out cubic error^2 x1.336163; tariff +415.442 MF / +0.169848%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_crunch_cubic_sketch64_v1/audit_l13_saved_v1.py](code/lead/runs/astra_crunch_cubic_sketch64_v1/audit_l13_saved_v1.py)
- [code/lead/runs/astra_crunch_cubic_sketch64_v1/price_l13_v3.py](code/lead/runs/astra_crunch_cubic_sketch64_v1/price_l13_v3.py)
- [code/lead/runs/astra_crunch_physics_v1/actual_next_cubic64_v1/audit_saved_v1.py](code/lead/runs/astra_crunch_physics_v1/actual_next_cubic64_v1/audit_saved_v1.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py](code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4534](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4534)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4565](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4565)

<details>
<summary>All result files and records (6)</summary>

- [SAVED_RESULT_v1.json](SAVED_RESULT_v1.json) · result
- [CLOSURE_v1.json](CLOSURE_v1.json) · record
- [SAVED_RESULT_v1-b3056e4e.json](SAVED_RESULT_v1-b3056e4e.json) · result
- [CLOSURE_v1-63e42978.json](CLOSURE_v1-63e42978.json) · record
- [TARIFF_v3.json](TARIFF_v3.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-signed-cubic-sketch-columns-layer-13-join-64-384) · [Copy provenance](../../../PROVENANCE.json)

