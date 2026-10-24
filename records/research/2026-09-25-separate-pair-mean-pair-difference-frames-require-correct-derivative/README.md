# Separate pair-mean and pair-difference frames require the correct derivative contraction

2026-09-25T19:42Z · finding · inconclusive · derived

Only reconstruction and the inner third-moment read scale with the old-tier frame rank. The paper cost model requires a separate-frame contraction; a naive expression would increase the bill, so it cannot justify implementation savings.

Recorded measurement:

      NEW HYPOTHESIS (lead -> covhunter): SPLIT u/v FRAMES FOR THE WHOLE OLD TIER. At each join, keep u columns in a rank-r_u frame and v columns in a rank-r_v frame. At L10 the cut drops 0.01% of u energy but 0.4% of v energy (§35), so r_u can go far below 384. Old-tier cost (27.7% of the bill) scales with (r_u + r_v)/768: at r_u = 256, about -4.6% of the bill before overlap with the L10 cut. Torch screen: r_u in {256, 192} at the S28 schedule. If raw is under +1%, run the full VALID40 and scope a native build (S30).
    - 2026-09-25 13:05 lead: SPLIT u/v FRAMES, NATIVE SCOPE (engineer, paper, S28 net 100): only reconstruct U@F (35.0 GF) + the inner read (28.3 GF) scale with rank = 21% of the bill (the 87 GF in _shared_old384_push are exact recent-cohort pushes). Estimate -2.0..-2.5% at r_u = 256, -4.0..-4.5% at r_u = 192, after the frame push and two joins. The inner MUST use the per-frame form lambda[2uv F_u + (u^2 + v^2/4) F_v] (cost (r_u + r_v)/(2r)); the naive form is a +5.6% LOSS. Engineer2 prototypes the native bill (2 nets, r_u 256/192) from this scope; covhunter screens raw on torch (r_u 256/192). A build follows only if raw at 192 or 256 leaves adjusted at least -1%.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1226](../../TEAM_LOG_2026-09-22_to_09-26.md#L1226)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1248](../../TEAM_LOG_2026-09-22_to_09-26.md#L1248)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-separate-pair-mean-pair-difference-frames-require-correct-derivative) · [Copy provenance](../../../PROVENANCE.json)

