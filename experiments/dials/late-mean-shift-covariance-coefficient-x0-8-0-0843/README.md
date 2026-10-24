# Late mean-shift covariance coefficient x0.8: -0.0843% adjusted; x0.6 reverses the gain

2026-10-20T17:07:24Z · experiment · rejected · measured

Reducing the late rank-two covariance coefficient by 20% gave a small 32-network gain and negative held-out result. Further reduction to x0.6 was worse.

Recorded measurement:

    - [fable-lead] 2026-10-20T17:07:24Z kc_late (kappa_C fraction of the GNN rank-2 covariance update at L8-14) on 32 ranks: x0.8 = adjusted -0.083% (se 0.074, lower 19/32, heldout25 see verdicts_v1.sh), x1.2 = +0.075%: a linear slope (less late re-add of the common mode is better), consistent with the agent's complement prediction (KC late with edge late). First sub-bar piece of the day; kc_late_mm (x0.6) and kc_late_mmm (x0.4) built for the lane after the joint FD. Joint FD started 17:06:52Z (3 processes, 41 settings x 2 nets each).
    AUDIT: lead FD glob included crash logs (fixed to exact A/B/C + settings validation); step units differ (lead x must /0.2 before builder); old agent setting composer drops iso and attenuates previous dials (use clean helper/builder). Builder now uses cache-safe joint_sci_v2; release wrapper extraction fixed. Fresh harness v5 warmed only first net: repeated no-op settings expose -0.31..-0.37% spurious bill on second nets; v6 warms every net and prints 12-digit U. Provisional half-profile6: corrected adjusted +0.028% (3/6 lower), not a win. Existing compare32 reports percentage-point deltas against95c; new compare_relative_v1 gives true relative vs rel_v4a AND legacy. kc_late_m true adjusted -0.0843%, held25 -0.0962% (legacy -0.0830/-0.0937); below fire bar. Require both conventions <=-0.35 plus heldout negative.
    VERIFIED: kc_late_mm complete all32 adjusted+0.0532%, held25+0.0418% (reject; more aggressive kc_late_mmm demoted). Only older kc_late_m is negative (-0.0843/-0.0962), below bar. Native4 fresh currentcompactJSK NT: p32/q128 -0.4323% SE0.652; p64/q192 -1.1976 SE0.889; p32/q64 -0.5382 SE0.527; p64/q128 -0.9011 SE0.828. All32 first7 for p32/q128 +1.180%, p64/q128 +1.614%; finish held25, do not bank fresh gains. Gram fourfresh -0.33155% SE0.1701 (raw-.2738,bill-.0579), matvec -.0184% SE.2445; full32 pending. Head train8 LOO -0.3667% raw relative to NT64/q128, no heldout read/no adjusted-v4a claim.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_kc_late_m_sci_v1.py](code/candidate_kc_late_m_sci_v1.py)
- [code/candidate_kc_late_mm_sci_v1.py](code/candidate_kc_late_mm_sci_v1.py)
- [code/candidate_kc_late_mmm_sci_v1.py](code/candidate_kc_late_mmm_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6207](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6207)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6218](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6218)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6228](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6228)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-late-mean-shift-covariance-coefficient-x0-8-0-0843) · [Copy provenance](../../../PROVENANCE.json)

