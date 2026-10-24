# Two-strip covariance safety variant of #337259: +0.429% adjusted on 32 networks; gates passed

2026-10-20T10:29:48Z · experiment · rejected · measured

The two-strip covariance variant traded +0.405% FLOPs and +0.026% MSE for runtime margin. It passed release gates but was held and never submitted.

Recorded measurement:

    [fable-lead] 2026-10-20T10:29:48Z SAFE TWIN JUDGED on the #337259 physics (32 ranks vs science32_rel_v4a_m15; not fired): twin_v4a all32 adjusted +0.429% (se 0.071, lower 4/32) = bill +0.405% + raw +0.026%; heldout25 +0.363% (raw -0.041%, bill +0.407%). Platform projection ~1.4346e-9 if the bill transfers in full (still 4th, above Luna 1.484). Its value is margin only (10-19 read: requests -1,926/net, residual x0.95). Gates next (release_min_v1.sh on candidate_twin_v4a.py 87d4feb5, REF = the #337259 archive).
    [fable-lead] 2026-10-20T10:09:21Z SAFE TWIN PREPARED ON THE #337259 PHYSICS (not fired; Matt's word required): candidate_twin_v4a.py (87d4feb5, release form = candidate_rel_v4a.py + the two-strip L6-15 fragment of candidate_jsym_twostrip_L6_15_v2 appended last) and the science copy candidate_twin_v4a_sci_v1.py (663539c2). Native smoke on net 35: FINAL 10.9844e-9 vs base 11.0143e-9 (-0.27% raw on this net; the strips change kernel order). 32-rank judge lane twin_v4a launched (queue_v7) vs science32_rel_v4a_m15 for the adjusted cost (10-19 read on the old base: raw -0.12%, bill +0.26%, requests -1,926/net, residual x0.95); gates after the judge. Purpose: a fire-ready second designated file for the private evaluation the moment Matt decides.
    [fable-lead] 2026-10-20T10:30:11Z SAFE TWIN GATES CLEAR (HELD, not fired; Matt's word required): fable_twin_v4a_87d4feb5_v1 -> setup5 PASS (max start envelope 3.31 s of 5), stage A PASS (archive release/runs/fable_twin_v4a_87d4feb5_v1/submission.tar.gz sha 2e245e1d4b0f, 714,239 bytes, 23 einsum signatures, none new vs proven), prefire clear except the approved gc import. Fire recipe if authorized: fire/fire_once.py on that archive with a description naming it the private-margin twin of #337259 (+0.43% adjusted locally, requests -1,926/net), then fire/poll_eval.sh (tag relative to fire/).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_twin_v4a.py](code/candidate_twin_v4a.py)
- [code/candidate_twin_v4a_sci_v1.py](code/candidate_twin_v4a_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6177](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6177)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6175](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6175)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6179](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6179)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [validate.json](validate.json) · result
- [stage_a.json](stage_a.json) · result
- [package.json](package.json) · result
- [validate_package.json](validate_package.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-two-strip-covariance-safety-variant-337259-0-429-adjusted) · [Copy provenance](../../../PROVENANCE.json)

