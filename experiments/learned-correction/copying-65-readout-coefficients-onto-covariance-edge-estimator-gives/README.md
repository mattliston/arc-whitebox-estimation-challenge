# Copying 65 readout coefficients onto the covariance-edge estimator gives eight-network adjusted ratio 0.995858110143

2026-09-21 · experiment · adopted · measured

The official bank #331684 (full lower-moment bank plus the fixed covariance-edge kernel) and the accepted packed readout #331682 (same bank with its last 65 learned final-head values replaced) were two branches of #331672 that had each improved the common ancestor separately; their fixed composition copied only the 64 final-head weights and the final bias from #331682 into #331684, with no fit and all other packed bytes identical (inverse edits restore the whole bank source). On a sealed roster of eight reused research networks outside both incremental fits it passed its gate, and the ordinary NumPy-free client reproduced all 16 native output bytes with the bank's complete bills.

Recorded measurement:

    pooled adjusted/bank 0.995858110143 (0.4142% reduction), 5/8 wins, worst 1.004219075102; parent pooled adjusted 3.6242291716621936e-09 vs composition 3.609218013617435e-09; per-network ratios 37: 0.998019431551, 39: 0.981771106430, 42: 0.990562483215, 48: 0.994769747959, 59: 1.003515583802, 63: 0.989532289242, 66: 1.004219075102, 67: 1.001106022010; client pooled wall ratio 1.001256078524, worst pair 1.016428910214, max residual 0.209131764 s; 28 full calls, 11540802585885 FLOPs

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/frozen_covariance_readout_composition_v1/candidate.py](code/frozen_covariance_readout_composition_v1/candidate.py)
- [code/frozen_covariance_readout_composition_v1/build.py](code/frozen_covariance_readout_composition_v1/build.py)
- [code/frozen_covariance_readout_composition_v1/run.py](code/frozen_covariance_readout_composition_v1/run.py)
- [code/frozen_covariance_readout_composition_v1/native.py](code/frozen_covariance_readout_composition_v1/native.py)
- [code/frozen_covariance_readout_composition_v1/score.py](code/frozen_covariance_readout_composition_v1/score.py)
- [code/frozen_covariance_readout_composition_v1/unit.py](code/frozen_covariance_readout_composition_v1/unit.py)
- [code/frozen_covariance_readout_composition_v1/common.py](code/frozen_covariance_readout_composition_v1/common.py)
- [code/frozen_covariance_readout_composition_v1/qualify.py](code/frozen_covariance_readout_composition_v1/qualify.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/readout_residual_runtime_v1/candidate.py](code/readout_residual_runtime_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)

## Evidence

- [ACCURACY_RESULT.md#L3](ACCURACY_RESULT.md#L3)
- [ACCURACY_RESULT.md#L20](ACCURACY_RESULT.md#L20)
- [SOURCE_REVIEW.md#L3](SOURCE_REVIEW.md#L3)
- [REEXAMINATION_INDEX_20260919.md#L609](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L609)
- [REEXAMINATION_INDEX_20260919.md#L611](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L611)

<details>
<summary>All result files and records (5)</summary>

- [ACCURACY_RESULT.md](ACCURACY_RESULT.md) · result
- [READY_RESULT.md](READY_RESULT.md) · result
- [SOURCE_REVIEW.md](SOURCE_REVIEW.md) · record
- [TIMING_SCOPE.md](TIMING_SCOPE.md) · record
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- ← measured by: [Automatic approval review rejected the readout-composition upload before process creation; renewed approval requested](../../../records/research/2026-09-21-automatic-approval-review-rejected-readout-composition-upload-before-process/README.md)
- enables: [Submission #331682: lossless static parameter packing of the 65-coefficient final-head refit; 100/100 completed](../../../submissions/331682/README.md)
- enables: [Submission #331684: packed frozen covariance-edge correction at layers 8-14 with full lower-moment support](../../../submissions/331684/README.md)
- shipped in: [Submission #331759: copied 65-coefficient readout on the covariance-edge estimator; 11 of 100 networks failed](../../../submissions/331759/README.md)
- ← shipped in: [Submission #331759: copied 65-coefficient readout on the covariance-edge estimator; 11 of 100 networks failed](../../../submissions/331759/README.md)

[Research graph](../../../graph/index.html#node=2026-09-21-copying-65-readout-coefficients-onto-covariance-edge-estimator-gives) · [Copy provenance](../../../PROVENANCE.json)

