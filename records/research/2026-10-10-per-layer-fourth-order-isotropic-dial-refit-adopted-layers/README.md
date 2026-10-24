# Per-layer fourth-order and isotropic dial refit adopted at layers 10-14

2026-10-10T21:02Z · decision · adopted · measured

Two accepted steps from joint global-dial and learned-array refit (jmH); the natively free form zero-extra-FLOP late-band refit (jmIf) keeps 55-60% of the untouched gain and late-band refit with isotropic per-layer shift (jmIv) (iso +0.030 at layers 10-14 through the per-layer switch per-layer literal switch (DLY@)) 80-85%, so candidate generation 46 (S46) = jmIv plus the push row cut at gate 1e-3.

Recorded measurement:

    jmI vs jmH: 48 clean -0.82% (se 0.12), 12 fresh -0.45%, 32 judging-only -0.61%; jmIv untouched 32 -0.66% (se 0.15, 25 of 32); jmIf -0.36%; jmJ (384 base) no step; jmL (edge arrays) no step

## Programs

- [candidate.py](candidate.py)
- [fragment_pld_v1.py](fragment_pld_v1.py)
- [read_pt.py](read_pt.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L674](../../TEAM_LOG_2026-10-01_to_10-22.md#L674)
- [TEAM_LOG_2026-10-01_to_10-22.md#L675](../../TEAM_LOG_2026-10-01_to_10-22.md#L675)
- [TEAM_LOG_2026-10-01_to_10-22.md#L676](../../TEAM_LOG_2026-10-01_to_10-22.md#L676)
- [TEAM_LOG_2026-10-01_to_10-22.md#L677](../../TEAM_LOG_2026-10-01_to_10-22.md#L677)
- [TEAM_LOG_2026-10-01_to_10-22.md#L679](../../TEAM_LOG_2026-10-01_to_10-22.md#L679)
- [TEAM_LOG_2026-10-01_to_10-22.md#L683](../../TEAM_LOG_2026-10-01_to_10-22.md#L683)
- [TEAM_LOG_2026-10-01_to_10-22.md#L684](../../TEAM_LOG_2026-10-01_to_10-22.md#L684)
- [TEAM_LOG_2026-10-01_to_10-22.md#L686](../../TEAM_LOG_2026-10-01_to_10-22.md#L686)
- [TEAM_LOG_2026-10-01_to_10-22.md#L688](../../TEAM_LOG_2026-10-01_to_10-22.md#L688)
- [TEAM_LOG_2026-10-01_to_10-22.md#L689](../../TEAM_LOG_2026-10-01_to_10-22.md#L689)
- [TEAM_LOG_2026-10-01_to_10-22.md#L705](../../TEAM_LOG_2026-10-01_to_10-22.md#L705)
- [TEAM_LOG_2026-10-01_to_10-22.md#L707](../../TEAM_LOG_2026-10-01_to_10-22.md#L707)
- [TEAM_LOG_2026-10-01_to_10-22.md#L711](../../TEAM_LOG_2026-10-01_to_10-22.md#L711)
- [TEAM_LOG_2026-10-01_to_10-22.md#L713](../../TEAM_LOG_2026-10-01_to_10-22.md#L713)
- [TEAM_LOG_2026-10-01_to_10-22.md#L753](../../TEAM_LOG_2026-10-01_to_10-22.md#L753)
- [TEAM_LOG_2026-10-01_to_10-22.md#L658](../../TEAM_LOG_2026-10-01_to_10-22.md#L658)
- [TEAM_LOG_2026-10-01_to_10-22.md#L673](../../TEAM_LOG_2026-10-01_to_10-22.md#L673)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Global dials near optimum after refit; remaining late-band signals require per-layer constants](../../../experiments/dials/global-dials-near-optimum-after-refit-remaining-late-band/README.md)
- shipped in: [Submission #335259: per-layer refit and a gate-floor 1e-3 dead-side push cut](../../../submissions/335259/README.md)
- ← enables: [Untouched generated networks become the primary judge because reused ranks overstate refit gains](../2026-10-10-untouched-generated-networks-become-primary-judge-because-reused-ranks/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers) · [Copy provenance](../../../PROVENANCE.json)

