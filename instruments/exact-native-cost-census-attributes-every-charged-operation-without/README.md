# Exact native cost census attributes every charged operation without changing predictions

2026-09-28 · tool · informational · measured

The unchanged rank35 estimator was instrumented after setup, with hooks restored afterward. Exact prediction and bill parity established the cost shares and showed that chunk partitioning alone did not offer a useful arithmetic saving.

Recorded measurement:

    Full bill276,549,645,399F;deadcut transport43.4479%,tensor read34.3864%,covariance-star11.0386%;inner-dimension split24.8960%;leaf unwind57.8568%;padding/output copies0.0263%.

## Programs

- [code/census.py](code/census.py)

## Evidence

- [HILL_CLIMB_before_uvp_bcache_v1.md#L25](../../experiments/birth-terms/birth-selector-enlarged-32-units-layers-110-expanded-adjusted/HILL_CLIMB_before_uvp_bcache_v1.md#L25)

<details>
<summary>All result files and records (1)</summary>

- [HILL_CLIMB_before_uvp_bcache_v1.md](../../experiments/birth-terms/birth-selector-enlarged-32-units-layers-110-expanded-adjusted/HILL_CLIMB_before_uvp_bcache_v1.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-28-exact-native-cost-census-attributes-every-charged-operation-without) · [Copy provenance](../../PROVENANCE.json)

