# Removing four unused materializations saves 646348799 FLOPs per network and improves adjusted error 0.13830087084589%

2026-09-08T08:49Z · experiment · adopted · measured

Removing unused newborn-tail materializations, off-diagonal innovation temporaries, a duplicate CDF and a duplicate feature square from the resident-Gram512 parent (SHA 0c21db7a) gave estimator 409f903e whose 16 served output layers are byte-identical to the parent on all 14 independent nets and the repeat, so the noise-corrected raw is inherited exactly while every complete bill is lower. It became the best fully qualified local component and the parent of the birth-direction fit; it was packaged but never uploaded on its own.

Recorded measurement:

    combined14 adjusted 4.770235434670105e-9 vs 4.7768418485455745e-9 (-0.13830087084589%), same NC raw 2.2458484596872894e-8, mean complete 466058731667.9286 F (saving 646348799 F/card); public8 5.183911640883259e-9 vs 5.191072991196909e-9, virgin6 4.218667159719234e-9 vs 4.2245336583437965e-9; native wall 18.1594-18.9677 s, peak RSS 7507623936 B; NumPy-free remote 35/100/500/3016500/35: wall 29.270754417-30.438686791 s, residual .11579291600000019-.13479798299999857 s, combined RSS 7488946176 B, native/remote drift 3.159046173095703e-6; same-backend combined4 adjusted 4.937998158721365e-9 vs 4.944830426071634e-9

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/unused_materializations_v1/build.py](code/unused_materializations_v1/build.py)
- [code/unused_materializations_v1/check.py](code/unused_materializations_v1/check.py)
- [code/unused_materializations_v1/check_remote.py](code/unused_materializations_v1/check_remote.py)
- [code/build_pair_polynomial_f32_candidate_v1.py](code/build_pair_polynomial_f32_candidate_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_pair_polynomial_f32_v1.py](code/candidate_response_pair_polynomial_f32_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)

<details>
<summary>Additional programs and dependencies (23)</summary>

- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_pair_polynomial_f32_dev_v1.py](code/check_pair_polynomial_f32_dev_v1.py)
- [code/check_pair_polynomial_f32_panel_v1.py](code/check_pair_polynomial_f32_panel_v1.py)
- [code/check_remote_series_v1.py](code/check_remote_series_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/check_response_rows_panel_v1.py](code/check_response_rows_panel_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/pair_polynomial_f32_source_v1.py](code/pair_polynomial_f32_source_v1.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)
- [code/resident_gram512_v1/estimator.py](code/resident_gram512_v1/estimator.py)
- [code/run_remote_series_v1.py](code/run_remote_series_v1.py)
- [code/terminal_dead_consumers_v1/check_panel.py](code/terminal_dead_consumers_v1/check_panel.py)
- [code/terminal_dead_consumers_v1/check_remote_v2.py](code/terminal_dead_consumers_v1/check_remote_v2.py)
- [code/terminal_dead_consumers_v1/check_synthetic.py](code/terminal_dead_consumers_v1/check_synthetic.py)
- [code/terminal_dead_consumers_v1/estimator.py](code/terminal_dead_consumers_v1/estimator.py)
- [code/terminal_dead_consumers_v1/source.py](code/terminal_dead_consumers_v1/source.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [QUALIFIED_2026_09_08_PDT.md#L13](QUALIFIED_2026_09_08_PDT.md#L13)
- [QUALIFIED_2026_09_08_PDT.md#L30](QUALIFIED_2026_09_08_PDT.md#L30)
- [QUALIFIED_2026_09_08_PDT.md#L34](QUALIFIED_2026_09_08_PDT.md#L34)
- [QUALIFIED_2026_09_08_PDT.md#L54](QUALIFIED_2026_09_08_PDT.md#L54)
- [QUALIFIED_2026_09_08_PDT.md#L65](QUALIFIED_2026_09_08_PDT.md#L65)
- [CONTINUATION_2026_09_08_0209_PDT.md#L10](CONTINUATION_2026_09_08_0209_PDT.md#L10)

<details>
<summary>All result files and records (4)</summary>

- [QUALIFIED_2026_09_08_PDT.md](QUALIFIED_2026_09_08_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · record
- [package_result.json](package_result.json) · result
- [CONTINUATION_2026_09_08_0209_PDT.md](CONTINUATION_2026_09_08_0209_PDT.md) · record

</details>

## Related work

- measured by: [Score arithmetic from the 4.777e-9 local reference: cost cuts alone cannot reach 3e-9, a raw-error route is mandatory](../../../records/research/2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts/README.md)
- ← enables: [Two unimplemented exact micro-cuts (denominator reuse, dead terminal covariance assignment): 70,284,316 F (0.0151%)](../../../records/research/2026-09-08-two-unimplemented-exact-micro-cuts-denominator-reuse-dead-terminal/README.md)

[Research graph](../../../graph/index.html#node=2026-09-08-removing-four-unused-materializations-saves-646348799-flops-per-network) · [Copy provenance](../../../PROVENANCE.json)

