# Linear-gate fourth-order columns transport only 5-16% of the true born contribution

2026-09-25T08:14Z · finding · informational · measured

The missing spread requires nonlinear dressing and conversion from third-order content. Increasing a linear column carrier rank cannot repair the wrong transport mechanism.

Recorded measurement:

    - 2026-09-25 01:20 analyst: k4 CARRIER COMPRESSIBILITY (fable_analyst_v1/RESULT.md section 37; with the physicist's CR-19 restarts; r35 + r100, MPS, gated). The "columns x layers x 2n^2" framing FAILS ON PHYSICS: born fourth-order content pushed with average-gate linear transport (the column-carrier model) carries only 5% (births L0), 7-8% (L1), 6-16% (L2-9) of those births' true contribution to per-unit k4 at L14, and in the per-unit spread its pattern is unaligned for L0-7 (cos -0.20..+0.15); no column count fixes that. Odd/even split (random-sign symmetrization after the birth ReLU, common random numbers): the early spread comes about half through born FOURTH-order content carried by the true ReLUs (coincident-index dressing, 5-10x the linear-gate value; 51-72% of the spread) and half through born THIRD-order content converting to k4 at later ReLUs (28-50%). Birth split: k4 NORM is early (L0 27%, L0-5 ~70%) but the per-unit SPREAD (23-26% of |T|, the spec's target) is born evenly (each 2-layer band 10-18%). So the spec needs a dressed full-depth fourth-order recursion plus k3->k4 conversion from the full K3 (closed family); plane floor alone ~+20-40% bill vs a -12..-18% lever. Cost on the physicist's page.
    - 2026-09-25 01:14 lead: ANALYST §37 (k4 carrier compressibility, r35+r100).
      - The per-unit k4 NORM at L13-14 is early-born (L0 27%, L0-5 ~70%). The per-unit SPREAD (what the card misses, 23-26% of |T|) is born evenly across layers.
      - Linear-gate column transport carries only 5-16% of born k4's true contribution at L14, with an unaligned spread pattern. No column count fixes that.
      - The early spread arrives ~half via born k4 dressed by the true ReLUs (5-10x the linear value) and ~half via born k3 converting to k4 at later ReLUs.
      The spec therefore needs a dressed full-depth fourth-order recursion plus k3->k4 conversion from the full K3 (the closed family). Confirms the physicist's cost closure.
    - 2026-09-25 01:15 [physicist] k3/k4 page (leadergap RESULT.md section N) now cites the analyst's §37 on both nets: column carrier reach 5-8% (L0-1) / 6-16% (L2-9) with no spread alignment for L0-7; early spread via born fourth-order content 50-72% and third-order conversion 28-50%; spread born evenly (10-18% per two-layer band). The analyst's +20-40% plane floor matches my 2.3-3.5 U/layer (+21-33%).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L851](../../TEAM_LOG_2026-09-22_to_09-26.md#L851)
- [TEAM_LOG_2026-09-22_to_09-26.md#L852](../../TEAM_LOG_2026-09-22_to_09-26.md#L852)
- [TEAM_LOG_2026-09-22_to_09-26.md#L853](../../TEAM_LOG_2026-09-22_to_09-26.md#L853)
- [TEAM_LOG_2026-09-22_to_09-26.md#L854](../../TEAM_LOG_2026-09-22_to_09-26.md#L854)
- [TEAM_LOG_2026-09-22_to_09-26.md#L855](../../TEAM_LOG_2026-09-22_to_09-26.md#L855)
- [TEAM_LOG_2026-09-22_to_09-26.md#L856](../../TEAM_LOG_2026-09-22_to_09-26.md#L856)
- [TEAM_LOG_2026-09-22_to_09-26.md#L857](../../TEAM_LOG_2026-09-22_to_09-26.md#L857)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT-1a40d4c3.md](../../../experiments/third-order-planes/gain-mixture-correction-half-strength-native-adjusted-ratio-1/RESULT-1a40d4c3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-linear-gate-fourth-order-columns-transport-only-5-16) · [Copy provenance](../../../PROVENANCE.json)

