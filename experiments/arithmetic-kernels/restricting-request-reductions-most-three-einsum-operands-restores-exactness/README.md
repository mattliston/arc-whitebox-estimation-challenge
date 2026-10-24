# Restricting request reductions to at most three einsum operands restores exactness

2026-09-25T10:30Z · experiment · adopted · measured

Removing the four-operand learned-network edit preserved all layer outputs and passed a numpy-free client census. The corrected package saved 766 requests and a small amount of work.

Recorded measurement:

    - 2026-09-25 03:30 [engineer2] GROUP 5 POST-MORTEM CONFIRMED (time-boxed, lock released): the ONLY non-identical piece was gnn_step's 4-operand einsum (flopscope runs 4+ operands pairwise; its rms is a scalar value in client mode). Corrected route B v2 = diet5/g5/build_g5_B_v2.py (bbd71a37) on the S26 pre-base a986fdc4 -> S26pre_g5B2_mem.py (cc44f7b7), no gnn_step edit, <= 3 operands, no scalar-value operands. Parity (setup first, 35/100/37/39): all 16 layers BITWISE identical, 0 signed-zero flips; bill -69M/net (-0.023%, removed recomputation). Numpy-free client census: clean, -766 requests/net (r100 29,365 -> 28,599; r35 30,017 -> 29,251), -97 free trips. SHELVED as ready; ships only on the lead's GO, after the porter's run_census_porter_v3.sh smoke.
    - 2026-09-25 03:31 lead: GROUP 5 POST-MORTEM (engineer2): the only cause was gnn_step's 4-operand einsum. Flopscope runs >= 4 operands pairwise (1-ulp differences), and rms arrives as a Python scalar in client mode. Route B v2 without that edit (build_g5_B_v2.py bbd71a37): bitwise on 4 nets, bill -0.023%, client census clean, -766 requests/net. HELD for S27 (S26 stays simple; the fragment was gated on the k-split v2 pre-base).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer2_v1/diet5/g5/build_g5_B_v2.py](code/fable_engineer2_v1/diet5/g5/build_g5_B_v2.py)
- [code/fable_engineer2_v1/diet5/g5/S26pre_g5B2_mem.py](code/fable_engineer2_v1/diet5/g5/S26pre_g5B2_mem.py)
- [code/fable_porter_v1/run_census_porter_v3.sh](code/fable_porter_v1/run_census_porter_v3.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L997](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L997)
- [TEAM_LOG_2026-09-22_to_09-26.md#L998](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L998)
- [FIRE_CHECKS.txt#L1](FIRE_CHECKS.txt#L1)
- [FIRE_CHECKS.txt#L4](FIRE_CHECKS.txt#L4)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record

</details>

## Related work

- shipped in: [Submission #332361: middle join rank 320 with refitted learned correction and matched dials](../../../submissions/332361/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-restricting-request-reductions-most-three-einsum-operands-restores-exactness) · [Copy provenance](../../../PROVENANCE.json)

