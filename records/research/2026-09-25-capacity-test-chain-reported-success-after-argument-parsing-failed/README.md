# Capacity-test chain reported success after argument parsing failed; result-line checks added

2026-09-25T10:16Z · incident · failed · measured

Both arms exited before training because a required device argument was missing, and the shell chain lost their exit status. The relaunch requires each arm own validation result line.

Recorded measurement:

    - 2026-09-25 03:16 lead: COVHUNTER CAPACITY TEST NEVER RAN. Both arms died in 2 s at 00:57 on argparse ("--device required"), and the chain logged "exit 0" (the exit status was lost). Relaunch requested: --device mps, fail-loud exit checks (pipefail/PIPESTATUS + require the arm's valid_epoch line), from loopR3 params (= S24) on the S24-structure card, MPS via the fixed lock. Bar: wide2x beats control by >= 0.5% VALID16. LESSON for all chains: success = the job's own result line, never just the shell's exit code.
    2026-09-25 03:19 covhunter: loopR3 JOINT REFIT VALID40 pre-gate (RESULT sec 32): (loopR3 params + exported dials) vs S23 torch, S23-structure card CPU: VALID40 -0.543% [CI -0.699,-0.387] 34/40 (VALID16 -0.52% 13/16, other24 -0.56% 21/24); rows loopR3_joint_vs_S23.json. CAPACITY TEST v3 relaunched 03:19 (arm A holds MPS; v2 had died on my chain bugs: lost REC line + exit check reading date's $?; v3 checks rc and the valid_epoch1 line).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L985](../../TEAM_LOG_2026-09-22_to_09-26.md#L985)
- [TEAM_LOG_2026-09-22_to_09-26.md#L986](../../TEAM_LOG_2026-09-22_to_09-26.md#L986)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [loopR3_joint_vs_S23.json](loopR3_joint_vs_S23.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-capacity-test-chain-reported-success-after-argument-parsing-failed) · [Copy provenance](../../../PROVENANCE.json)

