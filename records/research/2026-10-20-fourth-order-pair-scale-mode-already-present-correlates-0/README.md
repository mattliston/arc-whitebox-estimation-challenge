# Fourth-order pair scale mode is already present and correlates 0.967-0.982 with truth

2026-10-20T08:31:31Z · finding · informational · measured

The live pair map already carried the fourth-order pair scale-mode term. Its off-diagonal correlation with truth was 0.967-0.982 at layers 8-13, so the oracle changed a co-adapted recursion rather than supplying a missing term.

Recorded measurement:

      (3) The (2,2) plane is NOT missing: the live pair map carries it (cov += d d^T (Co^2 + k22/2)/2 with k22 = radial (v v^T + 2 C^2)/3), and that scale-mode object correlates 0.967-0.982 with the truth K22 off-diagonal at L8-13 (LS radial 0.040-0.061, rising with depth). The earlier k22_override oracle moved a co-adapted recursion (post22 -> radial -> next k22), not a missing term.
    [fable-lead] 2026-10-20T08:31:31Z FOUR OFFLINE CLOSURES on the late-layer error (nets 35/100, card quantities from the k21dump files at L8-14 vs the 32M mcall truth; lead/runs/fable_oct19_kc_v1/{dcov_structure,dv_regress,k22cov_offline,mehler_offline,edge2_offline}_v1.py):

## Programs

- [k22cov_offline_v1.py](k22cov_offline_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6150](../../TEAM_LOG_2026-10-01_to_10-22.md#L6150)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6147](../../TEAM_LOG_2026-10-01_to_10-22.md#L6147)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-fourth-order-pair-scale-mode-already-present-correlates-0) · [Copy provenance](../../../PROVENANCE.json)

