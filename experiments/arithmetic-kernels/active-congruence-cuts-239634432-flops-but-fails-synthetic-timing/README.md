# Active congruence cuts 239634432 FLOPs but fails its synthetic timing prerequisite

2026-09-08 · experiment · rejected · measured

A fixed active1024 congruence fixture reduced arithmetic while increasing wall time.

Recorded measurement:

    CLOSED_2026_09_07_PDT.md:
    | complete function FLOPs |2394771456|2155137024|
    | wall seconds |.13345454|.23416662|
    | relative Frobenius error to declared FP64 reference |1.99643155e-5|2.06238527e-5|
    | max absolute error |4.16556739e-6|3.80393051e-6|
    reference error and changed/parent discrepancy exceed the fixed2e-5
    gate. Strict cost reduction239634432F exactly matches source pricing,

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/active_congruence512_v1/check_synthetic.py](code/active_congruence512_v1/check_synthetic.py)
- [code/active_congruence512_v1/check_synthetic_v2.py](code/active_congruence512_v1/check_synthetic_v2.py)
- [code/active_congruence512_v1/congruence.py](code/active_congruence512_v1/congruence.py)
- [code/resident_gram512_v1/estimator.py](code/resident_gram512_v1/estimator.py)

## Evidence

- [CLOSED_2026_09_07_PDT.md#L3](CLOSED_2026_09_07_PDT.md#L3)
- [CLOSED_2026_09_07_PDT.md#L11](CLOSED_2026_09_07_PDT.md#L11)
- [CLOSED_2026_09_07_PDT.md#L12](CLOSED_2026_09_07_PDT.md#L12)
- [CLOSED_2026_09_07_PDT.md#L13](CLOSED_2026_09_07_PDT.md#L13)
- [CLOSED_2026_09_07_PDT.md#L14](CLOSED_2026_09_07_PDT.md#L14)

<details>
<summary>All result files and records (2)</summary>

- [CLOSED_2026_09_07_PDT.md](CLOSED_2026_09_07_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-active-congruence-cuts-239634432-flops-but-fails-synthetic-timing) · [Copy provenance](../../../PROVENANCE.json)

