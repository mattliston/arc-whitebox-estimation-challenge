# Final-layer error is mainly inherited: true preceding mean and covariance improve error 69.6-72.0%

2026-09-25T00:26Z · finding · informational · measured

Final-layer-born third- and fourth-cumulant errors explain a smaller part of the loss, with no useful scalar rescaling. The dominant error is inherited mean and covariance.

Recorded measurement:

    - 2026-09-24 17:26 analyst: LEVER 1 NO-GO (torch oracle, r35/r100, B2 ep1): truth mu15 -64.5/-57.5%; truth m14+C14 feeding L15 = truth mu+v = -72.0/-69.6% (INHERITED); L15-born k3+k4 -8.1/-7.0% (at the bar) but k3/k4 x(0.8..1.5) all worse (co-adapted optimum, no zero-cost handle); all L15 inputs truth -81/-74% (rest ~ 32M truth noise in mu15). J2W's final edge = better inherited m14/C14. RESULT.md sec 32. Starting lever 3.
    - 2026-09-24 17:45 lead: ANALYST L15 ORACLE (§32; 32M truth, r35/r100): truth m14+C14 feeding L15 = -72.0/-69.6%, the INHERITED part. Truth k3+k4 at L15 alone = -8.1/-7.0%, the BORN part: it is per-unit structure, since scalar k3/k4 rescales are co-adapted cliffs, so it needs per-unit-accurate k3/k4 at L15. Final-map + hook residual <= ~5-10%. The final-layer lever is NO-GO. J2W's final-layer edge is most likely better inherited m14/C14 (deep propagation). The physicist's per-unit k4 (cheap O(n^2) form) is the test of whether per-unit k4 structure is cheaply reachable. Analyst now on L3: LOO ridge on the top-32 C15 eigen-coefficients (40 TRAIN nets).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L511](../../TEAM_LOG_2026-09-22_to_09-26.md#L511)
- [TEAM_LOG_2026-09-22_to_09-26.md#L512](../../TEAM_LOG_2026-09-22_to_09-26.md#L512)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-final-layer-error-mainly-inherited-true-preceding-mean-covariance) · [Copy provenance](../../../PROVENANCE.json)

