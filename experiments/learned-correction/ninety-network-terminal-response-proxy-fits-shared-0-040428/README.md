# Ninety-network terminal-response proxy fits: shared +0.040428%, three-band +0.109454% leave-one-out

2026-10-21T03:43:19Z · experiment · rejected · measured

Two independently fixed terminal-response fits failed leave-one-network-out validation on 90 networks. Both were rejected before export or native candidate testing.

Recorded measurement:

    Current777a full-state capture completed90/90,30/30 originalguards at03:16:54Z (8:16:54 p.m.). Protected design3ade08b9 and exact weights/metadata preserved. Shared64 terminal response90LOO+0.040428% proxyMSE (49/90 lower), independently frozen3-band192+0.109454% (39/90): both reject beforeexport/native. Covariance-carrier responseaudit still72.759% relativeL2 error, so actual-native4D uses independently frozen geometry, not fitted proxy claims.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/truth_response95_v1/audit_covariance_response_v1.py](code/astra_oct20_bill_v1/truth_response95_v1/audit_covariance_response_v1.py)
- [code/astra_oct20_bill_v1/truth_response95_v1/fit_response90_threeband_v1.py](code/astra_oct20_bill_v1/truth_response95_v1/fit_response90_threeband_v1.py)
- [code/astra_oct20_bill_v1/truth_response95_v1/fit_response90_v1.py](code/astra_oct20_bill_v1/truth_response95_v1/fit_response90_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6511](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6511)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-ninety-network-terminal-response-proxy-fits-shared-0-040428) · [Copy provenance](../../../PROVENANCE.json)

