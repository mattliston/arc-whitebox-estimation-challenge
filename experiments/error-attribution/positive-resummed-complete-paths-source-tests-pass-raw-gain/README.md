# Positive-resummed complete paths: source tests pass, raw gain unproved

2026-09-06T02:43Z · experiment · informational · measured

Small dense certificates pass: positive component <6.7e-16 maximum error, 24-term primary-class relation <6.6e-19, arbitrary slice query <2.9e-17. - Rank35: weighted signal error ratio .3285296222; independent stream ratios .3512333266/.3400768351; cosine .8289957226. - Rank100 independent replication: weighted .3119823491; independent .3339069488/.3291934288; cosine .8472208080. Two new8M streams, about 1.42% observed weighted noise. This does not claim a final-layer score gain.

Recorded measurement:

    Small dense certificates pass: positive component <6.7e-16 maximum error, 24-term primary-class relation <6.6e-19, arbitrary slice query <2.9e-17. - Rank35: weighted signal error ratio .3285296222; independent stream ratios .3512333266/.3400768351; cosine .8289957226. - Rank100 independent replication: weighted .3119823491; independent .3339069488/.3291934288; cosine .8472208080. Two new8M streams, about 1.42% observed weighted noise. Both source gates pass. - Disabled modification reproduces the parent full plane bytewise on both nets; scalar-final disabled outputs also reproduce the parent. | net | unlearned new/parent | frozen bank heads new/parent | | --- | ---: | ---: | |35|1.01182495|1.07040151| |100|.97903039|1.15865814| This does not claim a final-layer score gain. The next action is the one predeclared20/4 matched control/source two-stage co-adaptation pilot in POSITIVE_PATH_MATCHED_PILOT_DESIGN_V1.md. No candidate or upload yet. Official usable bank remains329926, adjusted4.982231460183304e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/acquire_positive_path_replication_r100_v1.py](code/astra_phase2_reexamination_v1/acquire_positive_path_replication_r100_v1.py)
- [code/astra_phase2_reexamination_v1/analyze_positive_path_replication_r100_v1.py](code/astra_phase2_reexamination_v1/analyze_positive_path_replication_r100_v1.py)
- [code/astra_phase2_reexamination_v1/certify_positive_path_resummation_v1.py](code/astra_phase2_reexamination_v1/certify_positive_path_resummation_v1.py)
- [code/astra_phase2_reexamination_v1/check_cumulative_backward_source_v1.py](code/astra_phase2_reexamination_v1/check_cumulative_backward_source_v1.py)
- [code/astra_phase2_reexamination_v1/check_cumulative_positive_path_source_v1.py](code/astra_phase2_reexamination_v1/check_cumulative_positive_path_source_v1.py)
- [code/astra_phase2_reexamination_v1/cumulative_backward_source_v1.py](code/astra_phase2_reexamination_v1/cumulative_backward_source_v1.py)
- [code/astra_phase2_reexamination_v1/cumulative_positive_path_source_v1.py](code/astra_phase2_reexamination_v1/cumulative_positive_path_source_v1.py)
- [code/astra_phase2_reexamination_v1/exact_heavy_source_slices_v1.py](code/astra_phase2_reexamination_v1/exact_heavy_source_slices_v1.py)
- [code/astra_phase2_reexamination_v1/full_edge_gaussian_bond3_v1.py](code/astra_phase2_reexamination_v1/full_edge_gaussian_bond3_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py](code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py)
- [code/astra_phase2_reexamination_v1/positive_path_resummation_v1.py](code/astra_phase2_reexamination_v1/positive_path_resummation_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py](code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [POSITIVE_PATH_SOURCE_V1_VERDICT.md#L1](POSITIVE_PATH_SOURCE_V1_VERDICT.md#L1)
- [POSITIVE_PATH_SOURCE_V1_VERDICT.md#L12](POSITIVE_PATH_SOURCE_V1_VERDICT.md#L12)
- [POSITIVE_PATH_SOURCE_V1_VERDICT.md#L31](POSITIVE_PATH_SOURCE_V1_VERDICT.md#L31)

<details>
<summary>All result files and records (2)</summary>

- [POSITIVE_PATH_SOURCE_V1_VERDICT.md](POSITIVE_PATH_SOURCE_V1_VERDICT.md) · record
- [POSITIVE_PATH_MATCHED_PILOT_DESIGN_V1.md](POSITIVE_PATH_MATCHED_PILOT_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-positive-resummed-complete-paths-source-tests-pass-raw-gain) · [Copy provenance](../../../PROVENANCE.json)

