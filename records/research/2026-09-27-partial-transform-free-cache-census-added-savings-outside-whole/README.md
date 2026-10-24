# Partial-transform free-cache census: added savings outside whole repeats are at most 0.001047%

2026-09-27 · finding · rejected · measured

A fixed4096-scalar block census priced partial-transform reuse without charging lookup, copying or storage. Its union with earlier whole-input repeats remained below the implementation gate.

Recorded measurement:

    Partial savings0.326720%/0.359323%;additional outside whole repeats0.001047%/0.000460%;non-double-counted union0.346794%/0.381678%;all816 transform shapes/orders/bills preserved.

## Programs

- [census.py](census.py)
- [census_v2.py](census_v2.py)
- [close_census.py](close_census.py)

## Evidence

- [RESULTS.md#L1238](../../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1238)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-partial-transform-free-cache-census-added-savings-outside-whole) · [Copy provenance](../../../PROVENANCE.json)

