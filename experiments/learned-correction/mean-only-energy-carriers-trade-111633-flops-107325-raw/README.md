# Mean-only energy carriers trade -.111633% FLOPs for +.107325% raw MSE

2026-10-21T02:37:30Z · experiment · rejected · measured

Removing centered rank-two energy residuals while preserving exact means yielded -.004428% adjusted overall and +.006531% held-out. The FLOP saving was repaid in accuracy.

Recorded measurement:

    Latest native closures vs422: K31 coeff64 SCI26f34f85 full32-0.016118%/held-0.041932%, raw+0.053217%,bill-0.069299%,SE0.106296,15/32lower. Full-network official roster price exactly matches bill; +868chargedcalls means added runtime risk, no meaningful piece, no sweep. Pair-energy JSK SCI23861f61 full32-0.063740%/held-0.125080%, raw-0.075665%,bill+0.011936%,SE0.227022,15/32lower; first7+0.155334%; too weak/noisy for bank. EnergyE2/E3mean-only SCI55df56dd full32-0.004428%/held+0.006531%, raw+0.107325%,bill-0.111633%,SE0.087069,17/32lower; accuracy spent saving. All completed12native guards PASS. Original energy attempt failed before any prediction because source path was absolute; v3 plan/new outputv2 fixed only metadata. Read-only grade plan later restored missing historical393 denominator; predictions/source/guards untouched.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_vm_frontier_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_energy23_mean_v2.py](code/build_energy23_mean_v2.py)
- [code/check_energy23_mean_v2.py](code/check_energy23_mean_v2.py)
- [code/check_frontier_tiny_v5.py](code/check_frontier_tiny_v5.py)
- [code/frontier_audit_snapshot_20261021T023041Z/energy23_mean_v1/candidate_msg8_energy23_mean_sci_v1.py](code/frontier_audit_snapshot_20261021T023041Z/energy23_mean_v1/candidate_msg8_energy23_mean_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6462](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6462)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-mean-only-energy-carriers-trade-111633-flops-107325-raw) · [Copy provenance](../../../PROVENANCE.json)

