# Whole-pair global projection after layer 7: 512 pairs cut compute 19.03% but increase raw error 6.06x

2026-09-07T10:06:46Z · experiment · rejected · measured

The complete-pair kernel was implemented as a fixed online projection on the retained response-row parent: at post-layer-7 the original CP cohorts 0..6 were compressed to 512 complete signed pairs (1024 physical columns) selected by FP64 lambda^2 x all-distinct pair norm, weights solved from nine FP32 cross-Grams and a nine-product FP64 selected Gram with a fixed 1e-3 ridge, and the repeated entries restored into the star. After a passing tiny original-cube certificate, byte-exact null insertion and parent parity, the unwrapped two-net development pilot (ranks 35/100 plus repeat 35, stored 1e9 truths) cut the complete bill by 19.025102873980828% but multiplied raw MSE by 6.064166515704869 and adjusted by 4.910285966411552, failing the gate (both nets improve, >=1% mean adjusted gain); the exact construction was closed without rank, cut, ridge, precision or head sweeps. The static census had modeled a 20.3355%/19.9064% saving on old geometries.

Recorded measurement:

    Rank 35 cold: parent raw 2.1099028082828634e-8 / 476670806147 F / adj 4.573526314361569e-9 vs projected 1.1133888419226767e-7 / 384352970841 F / 1.9460199341401736e-8; rank 100: 1.997488978344223e-8 / 473227916215 / 4.298579128245664e-9 vs 1.3774019320118412e-7 / 384826542393 / 2.4104375505956108e-8; mean adjusted 4.436052721303617e-9 vs 2.1782287423678924e-8; compute saving 90359604564 F; candidate native calls 18.1539204167-18.3440241674 s, peak RSS 5163843584 bytes; tiny v2 certificate passed both fixtures (cross32 relative 1.5500586052889743e-07).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/whole_pair_kernel_core_v1.py](code/whole_pair_kernel_core_v1.py)
- [code/qualify_whole_pair_kernel_tiny_v2.py](code/qualify_whole_pair_kernel_tiny_v2.py)
- [code/build_whole_pair_kernel_candidate_v2.py](code/build_whole_pair_kernel_candidate_v2.py)
- [code/check_whole_pair_kernel_fixed_dev_v1.py](code/check_whole_pair_kernel_fixed_dev_v1.py)
- [code/candidate_whole_pair_kernel_post7_k512_v1.py](code/candidate_whole_pair_kernel_post7_k512_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)

</details>

## Evidence

- [WHOLE_PAIR_KERNEL_FIXED_DEV_V1_CLOSED.md#L14](WHOLE_PAIR_KERNEL_FIXED_DEV_V1_CLOSED.md#L14)
- [WHOLE_PAIR_KERNEL_FIXED_DEV_V1_CLOSED.md#L20](WHOLE_PAIR_KERNEL_FIXED_DEV_V1_CLOSED.md#L20)
- [WHOLE_PAIR_KERNEL_STATIC_REVIEW_V1.md#L12](WHOLE_PAIR_KERNEL_STATIC_REVIEW_V1.md#L12)
- [CONTINUATION_2026_09_07_031448_PDT.md#L23](CONTINUATION_2026_09_07_031448_PDT.md#L23)

<details>
<summary>All result files and records (8)</summary>

- [WHOLE_PAIR_KERNEL_FIXED_DEV_V1_CLOSED.md](WHOLE_PAIR_KERNEL_FIXED_DEV_V1_CLOSED.md) · result
- [WHOLE_PAIR_KERNEL_FIXED_PILOT_PROTOCOL_V1.md](WHOLE_PAIR_KERNEL_FIXED_PILOT_PROTOCOL_V1.md) · record
- [WHOLE_PAIR_KERNEL_STATIC_REVIEW_V1.md](WHOLE_PAIR_KERNEL_STATIC_REVIEW_V1.md) · record
- [WHOLE_PAIR_KERNEL_ROOT_ADMISSION_V1.md](WHOLE_PAIR_KERNEL_ROOT_ADMISSION_V1.md) · record
- [WHOLE_PAIR_KERNEL_TINY_V2_PREEXECUTION.md](WHOLE_PAIR_KERNEL_TINY_V2_PREEXECUTION.md) · record
- [whole_pair_kernel_tiny_v2.json](whole_pair_kernel_tiny_v2.json) · result
- [result.json](result.json) · result
- [CONTINUATION_2026_09_07_031448_PDT.md](CONTINUATION_2026_09_07_031448_PDT.md) · record

</details>

## Related work

- ← enables: [Coefficient-weighted cubic tensor projection with an FP64 solve passed tiny algebra and native API gates](../../../instruments/coefficient-weighted-cubic-tensor-projection-fp64-solve-passed-tiny/README.md)
- ← enables: [Complete signed-pair cubic kernel reduces leading Gram work 25% and requires a 512-dimensional solve](../../../records/research/2026-09-07-complete-signed-pair-cubic-kernel-reduces-leading-gram-work/README.md)

[Research graph](../../../graph/index.html#node=2026-09-07-whole-pair-global-projection-after-layer-7-512-pairs) · [Copy provenance](../../../PROVENANCE.json)

