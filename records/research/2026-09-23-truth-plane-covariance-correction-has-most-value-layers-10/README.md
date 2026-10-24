# Truth-plane covariance correction has most value at layers 10-12

2026-09-23T07:29Z · finding · informational · measured

Layer-resolved substitutions attributed most recoverable covariance error to the (2,1) plane, with a smaller (3,1) contribution. Correction gains were concentrated in late middle layers and were sub-additive.

Recorded measurement:

    2026-09-23 00:29 covhunter: PER-LAYER INJECTION MAP (r35/r100, torch learned ON; RESULT.md sec 11-12). One-step orthogonal injection (truth m,C at l-1; P=I-hh'/|h|^2): K21 truth removes .62-.72 of ||P dC P||_off (its covariance change is 69-99% ORTHOGONAL), K31 .19-.30, K22 .03-.11, k3/k4 0, planes together .92-1.01. Closed loop, truth planes K21+K22+K31 cov-only perp eps1: L2-14 -23.8/-25.4% (reproduces lead), bands L2-5 -1.9/+2.1, L6-9 -11.4/-2.9, L10-12 -16.0/-16.1, L13-14 -5.2/-9.4; singles L5 -5.4/-6.0, L10 -8.6/-7.2, L12 -5.6/-5.5, L2-4 at the 8M plane noise (+3.4/+4.0 at L2). Value concentrated in L10-12 (2/3 of all-layer), sub-additive bands. S2 knobs: keep v3 constants (S2 refit gives the same; c0/gamma valley fragile).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L121](../../TEAM_LOG_2026-09-22_to_09-26.md#L121)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-truth-plane-covariance-correction-has-most-value-layers-10) · [Copy provenance](../../../PROVENANCE.json)

