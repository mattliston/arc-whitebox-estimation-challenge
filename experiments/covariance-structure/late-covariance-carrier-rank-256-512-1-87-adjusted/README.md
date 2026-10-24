# Late covariance carrier rank 256 to 512: +1.87% adjusted; two other rank changes were no-ops

2026-10-20T04:29:22Z · experiment · rejected · measured

Doubling late carrier rank reduced MSE by -1.06% at +2.94% FLOPs. Two other source tails changed constants unused during prediction and produced byte-identical outputs.

Recorded measurement:

    - [fable-lead] 2026-10-20T04:29:22Z CAPACITY ARMS (32 store ranks vs stk1_r79 ref): lcr_512 (late low-rank carrier 256->512) raw -1.06% for bill +2.94% = adjusted +1.87% DEAD (rate 0.36, the capacity law holds); slr_512 and uvp_384 arms are byte-for-byte NO-OPS as built (identical numbers to their base rel_mstack1: _SLR_RANK/_UVP_RU tail assignments are not read at predict time) - not retested (size knobs trade <= 1:1 by the law); r2_nprho_b2_p +0.24% vs its base. Lanes now: rel_v4a ref + pe34_late_p/m, pe3/pe4_late_p, then gnng15/gnnglate, k415, dcvfloor arms. Fire bar per Matt 04:4xZ: stack >= 0.005e-9 better than #337259 (>= -0.35% adjusted on 32 ranks vs rel_v4a).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_lcr_512_sci_v1.py](code/candidate_lcr_512_sci_v1.py)
- [code/candidate_slr_512_sci_v1.py](code/candidate_slr_512_sci_v1.py)
- [code/candidate_uvp_384_sci_v1.py](code/candidate_uvp_384_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6133](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6133)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-late-covariance-carrier-rank-256-512-1-87-adjusted) · [Copy provenance](../../../PROVENANCE.json)

