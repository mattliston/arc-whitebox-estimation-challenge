# Einsum-chain fusion of 14 sites: -171 requests byte-identical but +0.13% residual instructions, so no gain

2026-10-16T15:28:46Z · experiment · rejected · measured

Fusing 14 einsum chains in LR2 (two multiplies into one einsum with helper checks) kept predictions byte-identical and removed requests, but a client einsum costs about 80K instructions against about 45K for a multiply, so total residual instructions rose. Rule: a request diet helps residual only when it removes a request; folding ops into a heavier request cuts the count, not the time. The v1 build crashed the server because a bare float reached einsum at _gop2_gnn_step.

Recorded measurement:

    LR2 vs diet2b v2 (72ceaf82): predictions byte-identical on networks 35 and 100, flops -86K, requests -165 first / -171 warm, Python instructions per request x1.0058 / x1.0055, total residual instructions +0.18% / +0.13%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_passrel_v1/diet2b_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_diet2b_v2.py](code/build_diet2b_v2.py)
- [code/cand_diet2b_v2.py](code/cand_diet2b_v2.py)
- [code/compare_sites_v1.py](code/compare_sites_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4286](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4286)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4292](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4292)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-einsum-chain-fusion-14-sites-171-requests-byte-identical) · [Copy provenance](../../../PROVENANCE.json)

