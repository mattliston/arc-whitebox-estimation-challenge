# Depth-seven long-inner reads: full-estimator adjusted gain is only 0.288486%

2026-09-27 · experiment · rejected · measured

The fixed deeper read kernel failed primitive speed admission and then received a separately declared whole-program screen. Its small native gain did not reach the expansion threshold.

Recorded measurement:

    Primitive arithmetic-5.801%,CPU2.335x on4 shapes;einsum leaf1.039x slower;native TRAIN35/100 raw-0.031817%,bill-0.255171%,adjusted-0.288486%,1/2 wins;fixed expansion gate1.5%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/depth7_inner_reads_v1/assess_native.py](code/depth7_inner_reads_v1/assess_native.py)
- [code/depth7_inner_reads_v1/bench.py](code/depth7_inner_reads_v1/bench.py)
- [code/depth7_inner_reads_v1/build.py](code/depth7_inner_reads_v1/build.py)
- [code/depth7_inner_reads_v1/candidate.py](code/depth7_inner_reads_v1/candidate.py)
- [code/depth7_inner_reads_v1/evaluate.py](code/depth7_inner_reads_v1/evaluate.py)
- [code/depth7_inner_reads_v1/fragment.py](code/depth7_inner_reads_v1/fragment.py)
- [code/depth7_inner_reads_v1/leaf_backend.py](code/depth7_inner_reads_v1/leaf_backend.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [RESULTS.md#L1326](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1326)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [FULL_PROGRAM_FOLLOWUP.md](FULL_PROGRAM_FOLLOWUP.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-depth-seven-long-inner-reads-full-estimator-adjusted-gain) · [Copy provenance](../../../PROVENANCE.json)

