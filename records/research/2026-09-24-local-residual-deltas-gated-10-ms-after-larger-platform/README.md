# Local residual deltas are gated at 10 ms after larger platform amplification

2026-09-24T22:38Z · decision · informational · measured

New arithmetic changes must report a paired cold residual delta and preserve margin to the 0.4-second cap. Official results showed that amplification varied between runs.

Recorded measurement:

    - 2026-09-24 15:38 lead: FIRED S15 as #332207 (S14 + engineer2 altbasis, e313ab77). VALID40 vs S14 -1.161%, 40/40. 3-way residual gate PASS but TIGHTENING: cold +0.024 s vs S13 (rule +0.03), projected platform max ~0.30 s (S14 actual .276 + .017). RESIDUAL IS NOW A BUDGET: each new layer must report its cold residual delta, and S16's paired gate vs S15 must show <= +0.01 s cold. Next: S16 = S15 + batched depth-6 (a4a1e0f7; chain running).
    - 2026-09-24 16:15 lead: GRADED #332207 (S15) = 2.9042e-9, NEW BANK (-1.21% vs S14). 100/100 complete, 0 failed, roundtrip max 69.9 s. RESIDUAL MAX 0.309 s (S14 .276): the platform added +0.033 s, more than the local paired gate's +0.017 s vs S14. Calibration: local paired cold delta x ~2 = platform delta. The cap is 0.4 s. RULE NOW: any new layer's local paired cold delta must be <= +0.01 s (~+0.02 s platform). The engineer's residual-trimming (ranked client-side costs, free on bill) is a PRIORITY: we need cushion before S16 (+6 ms local cold ~ +12 ms platform -> ~0.32 s).
    - 2026-09-24 16:14 lead: FIRED S16 as #332210 (S15 + bb6, a4a1e0f7). VALID40 vs S15 -1.431%, 40/40. Residual cold +0.010 s local vs S15, projected platform max ~0.33 s. NO FURTHER RESIDUAL-ADDING LAYERS until the engineer's residual trim lands: every S17 layer must be residual-neutral or come with a trim.
    - 2026-09-24 16:52 lead: GRADED #332210 (S16) = 2.8604e-9, NEW BANK (-1.51% vs S15). 100/100, 0 failed, roundtrip max 74.7 s. RESIDUAL max .313 s (S15 .309): +.004 on the platform, better than the ~2x-local projection (local was +.010). Today's ladder: 3.0463 -> 2.9743 -> 2.9398 -> 2.9042 -> 2.8604 (-6.1%). 2e-9 needs -30%.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L448](../../TEAM_LOG_2026-09-22_to_09-26.md#L448)
- [TEAM_LOG_2026-09-22_to_09-26.md#L454](../../TEAM_LOG_2026-09-22_to_09-26.md#L454)
- [TEAM_LOG_2026-09-22_to_09-26.md#L460](../../TEAM_LOG_2026-09-22_to_09-26.md#L460)
- [TEAM_LOG_2026-09-22_to_09-26.md#L473](../../TEAM_LOG_2026-09-22_to_09-26.md#L473)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-local-residual-deltas-gated-10-ms-after-larger-platform) · [Copy provenance](../../../PROVENANCE.json)

