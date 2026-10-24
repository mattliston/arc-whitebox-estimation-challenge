# Four-bin angular-energy information: inconclusive gate, no advance

2026-09-06T20:28Z · experiment · informational · measured

Phase 2, rank 35, preactivation 13 only. Judge completed September 6, 2026 at 20:26:34.872122 UTC (13:26:34 PDT). Observed G/bank ratios: cross 9.59220908, A 5.26832871, B 4.68021713.

Recorded measurement:

    Phase 2, rank 35, preactivation 13 only. Judge completed September 6, 2026 at 20:26:34.872122 UTC (13:26:34 PDT). This is an oracle local-mean diagnostic, not a final-layer raw/adjusted score or deployable estimator. A: 156.185 s, peak RSS 1,665,548,288 bytes. B: 158.033 s, 1,586,200,576 bytes. Each was the only MPS job, nice 10, three CPU threads, fallback disabled. G is the weighted conditional Gaussian ReLU mean in the four fixed hidden-energy bins. The bank comparator is the original NumPy bank post-13 reference after the served maximum-zero/float32 transform, not a native-D4 byte-parity claim. Errors are against each stream's same-sample empirical true ReLU mean. | Local mean-error quantity | Bank | Four-bin oracle G | | --- | ---: | ---: | | A/B cross energy | 1.42635238844e-8 | 1.36818703285e-7 | | Ordinary stream A MSE | 2.60648956896e-8 | 1.37318438182e-7 | | Ordinary stream B MSE | 2.91880755922e-8 | 1.36606531371e-7 | | Half squared replica difference | 1.33629617565e-8 | 1.43781491530e-10 | Observed G/bank ratios: cross 9.59220908, A 5.26832871, B 4.68021713. All three accuracy inequalities fail their fixed .50/.65/.65 limits. However, the bank noise/cross ratio is .93686258, above the required .10. The mixture ratio is .01008036 and passes. The unchanged primary verdict is therefore INCONCLUSIVE_FIXED_ACQUISITION, not a resolved information-gate failure or a theorem excluding all conditioning. Do not relabel it as a pass or silently loosen its noise criterion. Neither material nor boundary-sensitive estimated plug-in-bias warning fired. Eight-block leave-one-block cross ratios ranged 8.9316–10.8075; the approximate jackknife standard error was 1.7348. These are diagnostics, not rigorous confidence bounds or removal of common float32/finite-sample bias. The single unconditional Gaussian oracle G0 has cross error 2.38859760023e-7, 1.74581219 times the four-bin error. Thus the observed conditioning does reduce Gaussianization error, but the literal four-bin model remains far from the bank. The physical omitted-third ratio 74.61795 is diagnostic only and does not decide the primary gate.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/acquire_cross_depth_energy_replay_v2.py](code/astra_phase2_reexamination_v1/acquire_cross_depth_energy_replay_v2.py)
- [code/astra_phase2_reexamination_v1/acquire_cross_depth_energy_v1.py](code/astra_phase2_reexamination_v1/acquire_cross_depth_energy_v1.py)
- [code/astra_phase2_reexamination_v1/acquire_residual_one_high_oracle_v1.py](code/astra_phase2_reexamination_v1/acquire_residual_one_high_oracle_v1.py)
- [code/astra_phase2_reexamination_v1/cross_depth_energy_statistics_v1.py](code/astra_phase2_reexamination_v1/cross_depth_energy_statistics_v1.py)
- [code/astra_phase2_reexamination_v1/judge_cross_depth_energy_replay_v2.py](code/astra_phase2_reexamination_v1/judge_cross_depth_energy_replay_v2.py)
- [code/astra_phase2_reexamination_v1/judge_cross_depth_energy_v1.py](code/astra_phase2_reexamination_v1/judge_cross_depth_energy_v1.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1/whestbench_0.16.1/whestbench/protocol.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1/whestbench_0.16.1/whestbench/protocol.py)

## Evidence

- [CROSS_DEPTH_ENERGY_FIXED_INFORMATION_V2_VERDICT.md#L1](CROSS_DEPTH_ENERGY_FIXED_INFORMATION_V2_VERDICT.md#L1)
- [CROSS_DEPTH_ENERGY_FIXED_INFORMATION_V2_VERDICT.md#L4](CROSS_DEPTH_ENERGY_FIXED_INFORMATION_V2_VERDICT.md#L4)
- [CROSS_DEPTH_ENERGY_FIXED_INFORMATION_V2_VERDICT.md#L21](CROSS_DEPTH_ENERGY_FIXED_INFORMATION_V2_VERDICT.md#L21)

<details>
<summary>All result files and records (2)</summary>

- [CROSS_DEPTH_ENERGY_FIXED_INFORMATION_V2_VERDICT.md](CROSS_DEPTH_ENERGY_FIXED_INFORMATION_V2_VERDICT.md) · record
- [CROSS_DEPTH_ENERGY_TECHNICAL_REPLAY_V2.md](CROSS_DEPTH_ENERGY_TECHNICAL_REPLAY_V2.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-four-bin-angular-energy-information-inconclusive-gate-no-advance) · [Copy provenance](../../../PROVENANCE.json)

