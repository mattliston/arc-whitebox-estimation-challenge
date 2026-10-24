# Post-ReLU support constraint s_i(w) >= 0 on the carried third tensor: exact query and repair, no violation on record

2026-09-08T07:55Z · finding · inconclusive · derived

For a nonnegative post-ReLU vector with mean m, covariance C and central third tensor T, every ray w must satisfy s_i(w) = m_i w^T C w + T(e_i,w,w) - (w^T C e_i)^2 / m_i >= 0; the coordinate query was written against the live CP pushes (factor bank, active star and repeated plane) and a deficit r has the minimum-Frobenius single-halfspace repair DeltaT[z^3] = (3r/G) z_i ((x^T z)^2 - sum_a x_a^2 z_a^2) realized by three cubes plus a coordinate-star subtraction. No inspected receipt showed a warm-coordinate violation on the necessary scale, appending three columns breaks the 32-lattice reader path (R+3), and no census was admitted.

Recorded measurement:

    census plus FP64 check subtotals: 637,414,304 F on the historical a832, R21568 geometry and 895,246,368 F on the post12 width bound a1024, R26n; three future full-width pushes and plane products 34593792 F; for PSD C the lower bound on T(e_i,w,w) is at most -m_i (w^T C w)/2 when m_i^2 >= 2 C_ii

## Evidence

- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L60](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L60)
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L101](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L101)
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L146](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L146)
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L161](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L161)

<details>
<summary>All result files and records (2)</summary>

- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md](../2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md) · record
- [CONTINUATION_2026_09_08_0105_PDT.md](../2026-09-08-approximate-border-rank-multiplication-bini-schonhage-smirnov-circuits-save/CONTINUATION_2026_09_08_0105_PDT.md) · record

</details>

## Related work

- ← supersedes: [Simultaneous terminal support repair by a Fejer step: exact ten-product reader, 0.78 TF total, break-even raw .598](../2026-09-08-simultaneous-terminal-support-repair-fejer-step-exact-ten-product/README.md)

[Research graph](../../../graph/index.html#node=2026-09-08-post-relu-support-constraint-s-i-w-0-carried) · [Copy provenance](../../../PROVENANCE.json)

