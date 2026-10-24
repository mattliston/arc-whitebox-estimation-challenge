# Backend Cholesky overflow predicate preserves all 43 native and RPC cases for +3 FLOPs

2026-10-21T07:24:24Z · experiment · informational · measured

The original inequalities and RuntimeError semantics were retained while the final Boolean materialization was replaced by an eager one-by-one float64 Cholesky failure. A one-ULP test expectation was corrected after both parent and candidate correctly accepted the rounded boundary.

Recorded measurement:

    SLR semantic replacement is now verified, not merely proposed. `D/runtime_slr_cholesky_semantics_v1/tail_slr_cholesky_v1.py` SHA5f6d252425364e05b04896ccf14945133236b07b593a33d859cbcb926bcb6597 retains every original max/product/limit predicate and replaces only final Python bool/all with FP64 one-by-one Cholesky. False singular input eagerly fails before any next op/fetch/context exit, even with unused result. Native and NumPy-free RPC each43 boundary/invalid/NaN/Inf cases preserve acceptance and original RuntimeError text; full valid-guard debit+3F, primitive4F replacesall1F. V1 checker expected rejection one ULP above limit/384, but rounded product equals limit: both sources correctly accepted; preserved failureguard202127/log. Corrected v2 adds two-ULP rejected case and retains rounded accepted boundary. Guard202845 PASS07:07:49Z,8.50s/313MB/zeroSwap/events. Receiptsnative1af18539/RPC074567e0/guarddac3c9da. No guard weakened, no deferred NaN output, SDK unchanged.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/runtime_slr_cholesky_semantics_v1/tail_slr_cholesky_v1.py](code/runtime_slr_cholesky_semantics_v1/tail_slr_cholesky_v1.py)
- [code/runtime_slr_cholesky_semantics_v1/check_slr_v1.py](code/runtime_slr_cholesky_semantics_v1/check_slr_v1.py)
- [code/runtime_slr_cholesky_semantics_v1/run_checks_v1.py](code/runtime_slr_cholesky_semantics_v1/run_checks_v1.py)
- [code/runtime_slr_cholesky_semantics_v1/check_slr_v2.py](code/runtime_slr_cholesky_semantics_v1/check_slr_v2.py)
- [code/runtime_slr_cholesky_semantics_v1/run_checks_v2.py](code/runtime_slr_cholesky_semantics_v1/run_checks_v2.py)
- [code/runtime_first128_dcr_slr_v1/tail_slr_cholesky_v1.py](code/runtime_first128_dcr_slr_v1/tail_slr_cholesky_v1.py)
- [code/runtime_first128_dcr_scatter_v1/candidate_first128_smoff_npxoff_dcrscatter_release_v1.py](code/runtime_first128_dcr_scatter_v1/candidate_first128_smoff_npxoff_dcrscatter_release_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6853](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6853)

<details>
<summary>All result files and records (9)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [runtime_slr_cholesky_semantics_v2_GUARD.json](runtime_slr_cholesky_semantics_v2_GUARD.json) · result
- [PREPARE_v2.json](PREPARE_v2.json) · result
- [CHECKS_LAUNCH_v1.json](CHECKS_LAUNCH_v1.json) · result
- [SLR_RPC_CHECK_v2.json](SLR_RPC_CHECK_v2.json) · result
- [SLR_NATIVE_CHECK_v2.json](SLR_NATIVE_CHECK_v2.json) · result
- [CHECKS_LAUNCH_v2.json](CHECKS_LAUNCH_v2.json) · result
- [runtime_slr_cholesky_semantics_v1_GUARD.json](runtime_slr_cholesky_semantics_v1_GUARD.json) · result
- [PREPARE_v1.json](PREPARE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-backend-cholesky-overflow-predicate-preserves-all-43-native-rpc) · [Copy provenance](../../../PROVENANCE.json)

