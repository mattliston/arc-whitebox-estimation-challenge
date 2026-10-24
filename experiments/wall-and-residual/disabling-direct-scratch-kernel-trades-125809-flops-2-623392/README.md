# Disabling direct scratch kernel trades +.125809% FLOPs for -2.623392% residual instructions

2026-10-21T05:07:36Z · experiment · rejected · measured

Three quiet predictions remained exact when restoring the grouped scratch kernel parent. Reversing an exact-copy optimization also bought smaller residual-instruction improvement at a FLOP debit; neither was included in the then-authorized candidate.

Recorded measurement:

    Runtime research complete: SMoff exact3 predictions, originalguards clean, residualinstructions-2.623392%, requests-1.993875%, bill+.125809%, wall-.038059%; setuprequests15977->14847. NPcopyreversal exact3, reusedcontroltimingdescriptive: instructions-.703998%,bill+.0564348%,wall-.491026%. Both are paidcompanions, neither included in authorizedb54. Root sessions16617/35970 complete/resumed77740. V now independently qualifies zero-bill compactSMDIRECT fixed-AP-only, no FG/pair bundle and noMaclaunch.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/analyze_quiet_v1.py](code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/analyze_quiet_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/build_v1.py](code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/build_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/candidate_smoff_science_v1.py](code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/candidate_smoff_science_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/tail_smoff_v1.py](code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/tail_smoff_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/analyze_quiet_v1.py](code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/analyze_quiet_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/build_v1.py](code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/build_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/candidate_npxoff_science_v1.py](code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/candidate_npxoff_science_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/tail_npxoff_v1.py](code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/tail_npxoff_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6639](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6639)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-disabling-direct-scratch-kernel-trades-125809-flops-2-623392) · [Copy provenance](../../../PROVENANCE.json)

