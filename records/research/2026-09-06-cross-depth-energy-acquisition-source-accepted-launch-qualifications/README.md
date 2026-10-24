# Cross-depth energy acquisition source accepted with launch qualifications

2026-09-06T20:10Z · finding · informational · measured

- Main stream seed, PCG64DXSM construction, float32 normal API, unchanged 65,536 batches and final4,608 batch match the historical source. The synthetic statistic certificate has its own local generator before sampling; it does not change the stream generator or draw sequence. - Same copied float32 weights,13 successive clamp(x@W,min=0) operations (post0 through post12), then the pre13 weight product. - Main stream seed, PCG64DXSM construction, float32 normal API, unchanged 65,536 batches and final4,608 batch match the historical source.

Recorded measurement:

    - Main stream seed, PCG64DXSM construction, float32 normal API, unchanged 65,536 batches and final4,608 batch match the historical source. The synthetic statistic certificate has its own local generator before sampling; it does not change the stream generator or draw sequence. - Same copied float32 weights,13 successive clamp(x@W,min=0) operations (post0 through post12), then the pre13 weight product. The norm copies only read already-computed activations4/8; memberships remain aligned with rows of the same Z batch. No alternate forward kernel is used. - Replay post-mean/next-mean, second, square and third products and the final cumulant-centering expression match the old source. Extra old tensor-slice products not needed for the stated replay are absent. Actual bit fidelity remains the explicit final byte-parity gate, not a promise from source inspection or identical algebra alone. - Manifest role/rank/seed, weight bytes, thresholds and bank/reference hashes are checked. Teacher-forced input moments are never used as a comparator: only documented rounded shifts are loaded, and those must match each prior stream's stored actual shifts. Protocol/source files are hashed before sampling and rechecked afterwards. - Block boundaries are applied only to CPU sufficient statistics after each original whole-batch forward pass. Fixed8x1M partition and total counts are checked exactly. Raw sums use float64 arithmetic on actual float32 Z, not a reverse float32 shift reconstruction. Conditional k-statistic corrections and separate biased empirical identity checks match the audited formulas. - Four-bin/block sums occupy1MiB. Float64 working chunks are bounded at 8192x1024; no trajectory or dense per-bin covariance store is created. Full float32 batch CPU transfers are still about256MiB each. The actual ru_maxrss guard is 2GiB, checked after every batch using high-water RSS, not a claim that all combined GPU/CPU memory stays below2GiB. - The helper imports no Torch, card or threadpoolctl. The acquisition checks forbidden threadpoolctl before and after importing Torch, fixes Torch threads at3, requires nice>=10, and accepts no seed/count/shape/ layer/bin threshold override. SciPy's Gaussian consumer is not called by the MPS sampling/statistic-certificate path. - Existing outputs are refused. Qualification failures preserve their recorded evidence and do not restart or add samples. No fit, panel, submission, or Phase1 path is present.

## Programs

- [acquire_cross_depth_energy_v1.py](acquire_cross_depth_energy_v1.py)
- [../../../experiments/error-attribution/missing-third-tensor-fourth-cumulant-residual-opportunity-gate-passes/code/astra_phase2_reexamination_v1/cross_depth_energy_statistics_v1.py](../../../experiments/error-attribution/missing-third-tensor-fourth-cumulant-residual-opportunity-gate-passes/code/astra_phase2_reexamination_v1/cross_depth_energy_statistics_v1.py)
- [../../../experiments/error-attribution/missing-third-tensor-fourth-cumulant-residual-opportunity-gate-passes/code/astra_phase2_reexamination_v1/acquire_residual_one_high_oracle_v1.py](../../../experiments/error-attribution/missing-third-tensor-fourth-cumulant-residual-opportunity-gate-passes/code/astra_phase2_reexamination_v1/acquire_residual_one_high_oracle_v1.py)
- [acquire_cross_depth_energy_replay_v2.py](acquire_cross_depth_energy_replay_v2.py)
- [judge_cross_depth_energy_replay_v2.py](judge_cross_depth_energy_replay_v2.py)
- [judge_cross_depth_energy_v1.py](judge_cross_depth_energy_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_features.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_features.py)
- [protocol.py](protocol.py)

## Evidence

- [CROSS_DEPTH_ENERGY_ACQUISITION_SOURCE_AUDIT_V1.md#L1](CROSS_DEPTH_ENERGY_ACQUISITION_SOURCE_AUDIT_V1.md#L1)

<details>
<summary>All result files and records (1)</summary>

- [CROSS_DEPTH_ENERGY_ACQUISITION_SOURCE_AUDIT_V1.md](CROSS_DEPTH_ENERGY_ACQUISITION_SOURCE_AUDIT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-cross-depth-energy-acquisition-source-accepted-launch-qualifications) · [Copy provenance](../../../PROVENANCE.json)

