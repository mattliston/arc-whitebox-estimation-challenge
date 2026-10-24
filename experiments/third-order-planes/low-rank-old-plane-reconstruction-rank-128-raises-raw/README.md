# Low-rank old-plane reconstruction at rank 128 raises raw error 90.1%

2026-09-25T04:34Z · experiment · rejected · measured

Even retaining about 99% of old-plane energy lost important signal. Newborn and common birth operands were too information-rich to enter the same low-rank experiment.

Recorded measurement:

    2026-09-24 21:34 [engineer2] LOW-RANK TOLERANCE SCREEN of S21 (eddbcfcd), E-62 recipe (range finder k=r+16, 1 power step, exact truncation to r), in process, learned ON, no refit, nets 37/39/35/100 vs 8M truth; files fable_engineer2_v1/lowrank/ (lr_spectra, lr_e62_test, lr_content, lr_summary). Class map by caller (dataflow census v2, S21 r100): young-square reads 23.3% (closed: E-66), newborn bank push 22.9%, old tier ~26.2% rank-dependent (reconstruct 11.2, inner 7.0, pushes 4.7, join tiles 1.8, W5 shared_basis 0.8, old-plane read 0.6), batched C w + star s1 8.9% (C w closed: E-63), covariance sandwich 3.5% (closed: E-63), W5 newborn_common 3.6%, low-rank star machinery 2.8% (plain-rate range finder + thin products).
      - [LR] NEWBORN BANK PUSH (22.9%): content = the transported state itself (ratio 1); energy E(32/64/128) median .73/.90/.977 (99% needs rank 579 early -> 108 late) -> not star-like, NOT TESTED (rule).
      - [LR] OLD TIER (rank 384; ~26.2% rank-dependent): content ||old plane||/||read plane|| median .43 (.23-.62); energy E(32/64/128) median .77/.93/.991 (star-like by energy) -> TESTED: r128 raw +90.1% (per net +97.6/+80.6/+93.4/+88.9%) at a potential cut <= -17.4% -> KILLED at the first rank (r64 1 net +383%; control at r384 -0.20%). 99% energy is not enough: the old tier carries ~43% of the read plane.
      - [LR] W5 NEWBORN_COMMON BASE (3.6%): content ||common||/||common + directions t|| median .91; E(128) median .928 -> not star-like, NOT TESTED.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/lowrank`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lr_content.py](code/lr_content.py)
- [code/lr_e62_test.py](code/lr_e62_test.py)
- [code/lr_spectra.py](code/lr_spectra.py)
- [code/lr_summary.py](code/lr_summary.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L655](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L655)
- [TEAM_LOG_2026-09-22_to_09-26.md#L656](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L656)
- [TEAM_LOG_2026-09-22_to_09-26.md#L657](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L657)
- [TEAM_LOG_2026-09-22_to_09-26.md#L658](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L658)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-low-rank-old-plane-reconstruction-rank-128-raises-raw) · [Copy provenance](../../../PROVENANCE.json)

