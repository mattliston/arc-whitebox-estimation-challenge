# Gram arithmetic plus float32 matrix-vector changes: -0.206095% adjusted on 32 networks

2026-10-20T19:22:39Z · experiment · adopted · measured

The Gram change alone gave -0.1158% adjusted; combining it with float32 matrix-vector arithmetic gave -0.206095%. The held-out 25-network result was -0.204864%.

Recorded measurement:

    COMPLETED32: Gram -0.1158% SE.0888, held25-.1071%; Gram+f32matvec micro combined -0.206095% SE.095667,23/32 lower, held25-.204864%. BankKC late_m -.0843%,held-.0962%. Fresh Gram -.3315 did NOT retain its full magnitude. Joint GSK/GK4 early-10% still running: first7+.1330% (fresh4-.1893 not banked). Old Gram+pair+KC+copy stack d03b2500 first7-.2825%, remainder running. Do not claim the optimistic ~-.86% sum as achieved.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_rel_micro_kclm_copyall_astra.py](code/candidate_rel_micro_kclm_copyall_astra.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6245](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6245)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337396: Mean and carrier precision with copy removal; two residual failures](../../../submissions/337396/README.md)

[Research graph](../../../graph/index.html#node=2026-10-20-gram-arithmetic-plus-float32-matrix-vector-changes-0-206095) · [Copy provenance](../../../PROVENANCE.json)

