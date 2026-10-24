# Four-level Winograd synthetic profiles attribute only 14.3377180171-17.0538357931% of time to matrix products

2026-09-08 · finding · informational · measured

On two fixed large shapes, transforms and other kernel work dominated time even though multiplication dominated arithmetic.

Recorded measurement:

    VERDICT_2026_09_08_PDT.md:
    PID95540, 11:12:16.039695--11:12:17.196459 UTC. `run()` wall was
    |1024,832,8192|8,417,538,304|524,226,560|0.1128722113 / 0.1065244591|17.0538357931% / 15.6864989942%|
    |832,8192,832|6,921,666,752|544,727,040|0.1210212931 / 0.1202292107|14.6676666850% / 14.3377180171%|
    Official329926 remains4.982231460183304e-9 adjusted. Qualified local14

## Programs

- [run_profile.py](run_profile.py)
- [../../../experiments/wall-and-residual/exact-winograd-optimization/code/astra_phase2_reexamination_v1/packed_arena_generic.py](../../../experiments/wall-and-residual/exact-winograd-optimization/code/astra_phase2_reexamination_v1/packed_arena_generic.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py)

## Evidence

- [VERDICT_2026_09_08_PDT.md#L3](VERDICT_2026_09_08_PDT.md#L3)
- [VERDICT_2026_09_08_PDT.md#L7](VERDICT_2026_09_08_PDT.md#L7)
- [VERDICT_2026_09_08_PDT.md#L21](VERDICT_2026_09_08_PDT.md#L21)
- [VERDICT_2026_09_08_PDT.md#L22](VERDICT_2026_09_08_PDT.md#L22)
- [VERDICT_2026_09_08_PDT.md#L48](VERDICT_2026_09_08_PDT.md#L48)

<details>
<summary>All result files and records (3)</summary>

- [VERDICT_2026_09_08_PDT.md](VERDICT_2026_09_08_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [result.json](result.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-four-level-winograd-synthetic-profiles-attribute-only-14-3377180171) · [Copy provenance](../../../PROVENANCE.json)

