# Corrected body request package passes exactness and client smoke on the final composition

2026-09-25T18:50Z · experiment · adopted · measured

The three-or-fewer-operand package remains bitwise exact with the updated splitting and reuse stages. It removes hundreds of requests per network with a small work reduction.

Recorded measurement:

    - 2026-09-25 11:50 [engineer2] GROUP 5 v2 FOR S28 PASSES PARITY BITWISE (build_g5_B_v2.py, fragment b769aa28, on s26w_pre 9ef1a831 with treuse v2 75aa9233 appended last to both sides; diet5/g5/s28v2/): all 16 outputs bitwise equal on nets 35/100/37/39, 0 signed-zero flips, MSE ratio 1.0. Bill -69.06 MF on every net (-0.0226%). Requests -684 (net 35) and -734 (net 100) in the client census. Einsum gate PASS (0 new repeated-operand sites), numpy-free client smoke PASS. The chain's paired user-CPU and residual steps were stopped at 11:46 and the lock released, so S27t's smoke has it. Those two pairs can run after S27t's VALID40, or S28's own residual gate and VALID40 cover them, as for v5w (lead's call). Correction to my lock record: the chain took the lock at 11:23:59 and then waited 14 min on the heavy gate, the order my LOCK-ORDER LAW forbids; the next chain takes the machine gate first.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/diet5/g5`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_g5_B_v2.py](code/build_g5_B_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1183](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1183)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332361: middle join rank 320 with refitted learned correction and matched dials](../../../submissions/332361/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-corrected-body-request-package-passes-exactness-client-smoke-final) · [Copy provenance](../../../PROVENANCE.json)

