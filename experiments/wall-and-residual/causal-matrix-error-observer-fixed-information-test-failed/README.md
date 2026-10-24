# Causal matrix-error observer: fixed information test failed

2026-09-06T09:20Z · experiment · rejected · measured

The fixed read-only Gaussian-tangent error memory passed its prerequisites: synthetic tangent errors <=7.43e-10; exact linearity <=3.66e-16; small MPS/NumPy output <=2.37e-7 and gradient <=3.03e-7 relative. Current rank35/100 zero observer preserved the actual scalar-final P/Q bank byte-exactly. Pooled memory raw2.0567774985e-8, bank2.1201656114e-8, local2.1240007568e-8.

Recorded measurement:

    The fixed read-only Gaussian-tangent error memory passed its prerequisites: synthetic tangent errors <=7.43e-10; exact linearity <=3.66e-16; small MPS/NumPy output <=2.37e-7 and gradient <=3.03e-7 relative. Current rank35/100 zero observer preserved the actual scalar-final P/Q bank byte-exactly. Fixed nonzero wide MPS/NumPy relative errors <=9.53e-7. Twenty-four final rows were captured without targets in80.31s using two nice10 CPU workers. All15 prefix feature/sigma fields and existing final prediction/host/sigma/delta/h matched byte-for-byte. The actual scalar-final x and sigma replace the old full-final tape entries; other tape fields stay unchanged and rounded as originally stored. Final C is not consumed by the final observer response. Both locked100-epoch fits completed,87.61s total on one MPS worker, peak host RSS3,564,699,648 bytes. Existing1e9 final means only,20train and 37/39/42/48 held out. Frozen bank and main P/Q never modified. | Holdout | Bank raw MSE | Matrix memory / bank | Local control / bank | |---|---:|---:|---:| |37|2.1281180954e-8|.9502870722|.9924554326| |39|1.7861628910e-8|1.0088789229|1.0139762581| |42|2.3224433217e-8|.9811457271|1.0058288623| |48|2.2439381374e-8|.9465989321|.9968337969| Pooled memory raw2.0567774985e-8, bank2.1201656114e-8, local2.1240007568e-8. Memory/bank=.9701022823; memory/local=.9683506430;3/4 wins. Required<=.75,<=.85,>=3 wins. FAIL. This3% raw gain does not establish an adjusted gain and is too small for the extra full-matrix recurrence. No200-net fit, optimizer/feature/ source/window/scale sweep, state write, native port or bank follows. This is a result for the fixed fit, not an exact linear-representation ceiling or an impossibility claim about every matrix observer.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/matrix_error_observer_data_v1.py](code/matrix_error_observer_data_v1.py)
- [code/matrix_error_observer_torch_v1.py](code/matrix_error_observer_torch_v1.py)
- [code/matrix_error_observer_v1.py](code/matrix_error_observer_v1.py)
- [code/train_matrix_error_observer_v1.py](code/train_matrix_error_observer_v1.py)

## Evidence

- [MATRIX_ERROR_OBSERVER_V1_CLOSED.md#L1](MATRIX_ERROR_OBSERVER_V1_CLOSED.md#L1)
- [MATRIX_ERROR_OBSERVER_V1_CLOSED.md#L7](MATRIX_ERROR_OBSERVER_V1_CLOSED.md#L7)
- [MATRIX_ERROR_OBSERVER_V1_CLOSED.md#L17](MATRIX_ERROR_OBSERVER_V1_CLOSED.md#L17)
- [MATRIX_ERROR_OBSERVER_V1_CLOSED.md#L24](MATRIX_ERROR_OBSERVER_V1_CLOSED.md#L24)

<details>
<summary>All result files and records (3)</summary>

- [MATRIX_ERROR_OBSERVER_V1_CLOSED.md](MATRIX_ERROR_OBSERVER_V1_CLOSED.md) · record
- [matrix_error_observer_training_verdict_v1.json](matrix_error_observer_training_verdict_v1.json) · result
- [MATRIX_ERROR_OBSERVER_DESIGN_V1.md](MATRIX_ERROR_OBSERVER_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-causal-matrix-error-observer-fixed-information-test-failed) · [Copy provenance](../../../PROVENANCE.json)

