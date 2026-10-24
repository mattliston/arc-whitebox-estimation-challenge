# Sixth-cumulant (4,2) and (5,1) pair inheritance: residual time 0.5109498272649944 s failed

2026-08-27 · experiment · rejected · measured

The singleton target-blind rank-201 K42/K24 plus K51/K15 source/action transaction ran exactly once under the frozen V3 A1-B-A2 no-retry authority and atomically committed only combined_raw_k6.npy, its result, and manifest. Thirty-three of 34 gates pass: exact five-component public parity; every centered-Q7, partition, orientation, factor-15/factor-6, Gaussian-null, finite/nonzero, dtype, zero-diagonal, source-call, operation-call, total-FLOP, resident-set size (RSS), coordinate, and firewall gate; and equal measured deltas of 151,825,424,384 FLOPs, exactly the independent integrated prediction and below the 152,200,000,000 cap. The sole failure is instrumented B residual 0.5109498272649944 seconds versus the strict below-0.4-second gate (A1 0.35438148537650704; A2 0.3446839735843241).

Recorded measurement:

    The singleton target-blind rank-201 K42/K24 plus K51/K15 source/action transaction ran exactly once under the frozen V3 A1-B-A2 no-retry authority and atomically committed only combined_raw_k6.npy, its result, and manifest. Thirty-three of 34 gates pass: exact five-component public parity; every centered-Q7, partition, orientation, factor-15/factor-6, Gaussian-null, finite/nonzero, dtype, zero-diagonal, source-call, operation-call, total-FLOP, RSS, coordinate, and firewall gate; and equal measured deltas of 151,825,424,384 FLOPs, exactly the independent integrated prediction and below the 152,200,000,000 cap. The sole failure is instrumented B residual 0.5109498272649944 seconds versus the strict below-0.4-second gate (A1 0.35438148537650704; A2 0.3446839735843241). A later static scalar-witness rewrite removes the repeated plain-NumPy plane scans and is arithmetically cap-safe, but it changes only the diagnostic sink after the frozen residual failure. It is archived fail-closed as a forbidden counterfactual, not a retry or successor.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_q24_k42_k51_pair_inheritance_v1/k42_k51_instrumented_source.py](code/public_q24_k42_k51_pair_inheritance_v1/k42_k51_instrumented_source.py)
- [code/public_q24_k42_k51_pair_inheritance_v1/source_action_runner_scaffold.py](code/public_q24_k42_k51_pair_inheritance_v1/source_action_runner_scaffold.py)
- [code/public_q24_k42_k51_pair_inheritance_v1/verify_static.py](code/public_q24_k42_k51_pair_inheritance_v1/verify_static.py)
- [code/public_q24_k42_k51_scalar_witness_counterfactual_v1/k42_k51_scalar_witness_source.py](code/public_q24_k42_k51_scalar_witness_counterfactual_v1/k42_k51_scalar_witness_source.py)
- [code/public_q24_k42_k51_scalar_witness_counterfactual_v1/counterfactual_runner_scaffold.py](code/public_q24_k42_k51_scalar_witness_counterfactual_v1/counterfactual_runner_scaffold.py)
- [code/public_q24_k42_k51_scalar_witness_counterfactual_v1/verify_static.py](code/public_q24_k42_k51_scalar_witness_counterfactual_v1/verify_static.py)
- [code/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py](code/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py)

## Evidence

- [DERIVATION.md#L1](DERIVATION.md#L1)
- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json#L1](K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json#L1)
- [STATIC_VERIFICATION_V2.json#L1](STATIC_VERIFICATION_V2.json#L1)
- [SOURCE_FREEZE_V2.json#L1](SOURCE_FREEZE_V2.json#L1)
- [STATIC_VERIFICATION_V3.json#L1](STATIC_VERIFICATION_V3.json#L1)
- [SOURCE_FREEZE_V3.json#L1](SOURCE_FREEZE_V3.json#L1)
- [K42_K51_INTEGRATED_ACCOUNTING_RECEIPT_V1.json#L1](K42_K51_INTEGRATED_ACCOUNTING_RECEIPT_V1.json#L1)
- [K42_K51_SOURCE_ACTION_RELEASE.json#L1](K42_K51_SOURCE_ACTION_RELEASE.json#L1)
- [K42_K51_SOURCE_ACTION_RANK0201_RUN.lock#L1](K42_K51_SOURCE_ACTION_RANK0201_RUN.lock#L1)
- [K42_K51_SOURCE_ACTION_RANK0201_A1_WORKER_CONSUMED.json#L1](K42_K51_SOURCE_ACTION_RANK0201_A1_WORKER_CONSUMED.json#L1)
- [K42_K51_SOURCE_ACTION_RANK0201_B_WORKER_CONSUMED.json#L1](K42_K51_SOURCE_ACTION_RANK0201_B_WORKER_CONSUMED.json#L1)
- [K42_K51_SOURCE_ACTION_RANK0201_A2_WORKER_CONSUMED.json#L1](K42_K51_SOURCE_ACTION_RANK0201_A2_WORKER_CONSUMED.json#L1)
- [COMMIT_MANIFEST.json#L1](COMMIT_MANIFEST.json#L1)
- [K42_K51_SOURCE_ACTION_RANK0201_RESULT.json#L2348](K42_K51_SOURCE_ACTION_RANK0201_RESULT.json#L2348)
- [K42_K51_RANK0201_TERMINAL_CLOSURE_RECEIPT.json#L317](K42_K51_RANK0201_TERMINAL_CLOSURE_RECEIPT.json#L317)
- [DERIVATION-08fcc6e0.md#L27](DERIVATION-08fcc6e0.md#L27)
- [PREREGISTRATION-e0732f21.json#L22](PREREGISTRATION-e0732f21.json#L22)
- [STATIC_VERIFICATION_V1.json#L453](STATIC_VERIFICATION_V1.json#L453)
- [SOURCE_FREEZE_V1.json#L1](SOURCE_FREEZE_V1.json#L1)

<details>
<summary>All result files and records (20)</summary>

- [DERIVATION.md](DERIVATION.md) · record
- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json](K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json) · result
- [STATIC_VERIFICATION_V2.json](STATIC_VERIFICATION_V2.json) · result
- [SOURCE_FREEZE_V2.json](SOURCE_FREEZE_V2.json) · result
- [STATIC_VERIFICATION_V3.json](STATIC_VERIFICATION_V3.json) · result
- [SOURCE_FREEZE_V3.json](SOURCE_FREEZE_V3.json) · result
- [K42_K51_INTEGRATED_ACCOUNTING_RECEIPT_V1.json](K42_K51_INTEGRATED_ACCOUNTING_RECEIPT_V1.json) · result
- [K42_K51_SOURCE_ACTION_RELEASE.json](K42_K51_SOURCE_ACTION_RELEASE.json) · result
- [K42_K51_SOURCE_ACTION_RANK0201_A1_WORKER_CONSUMED.json](K42_K51_SOURCE_ACTION_RANK0201_A1_WORKER_CONSUMED.json) · result
- [K42_K51_SOURCE_ACTION_RANK0201_B_WORKER_CONSUMED.json](K42_K51_SOURCE_ACTION_RANK0201_B_WORKER_CONSUMED.json) · result
- [K42_K51_SOURCE_ACTION_RANK0201_A2_WORKER_CONSUMED.json](K42_K51_SOURCE_ACTION_RANK0201_A2_WORKER_CONSUMED.json) · result
- [COMMIT_MANIFEST.json](COMMIT_MANIFEST.json) · result
- [K42_K51_SOURCE_ACTION_RANK0201_RESULT.json](K42_K51_SOURCE_ACTION_RANK0201_RESULT.json) · result
- [K42_K51_RANK0201_TERMINAL_CLOSURE_RECEIPT.json](K42_K51_RANK0201_TERMINAL_CLOSURE_RECEIPT.json) · result
- [DERIVATION-08fcc6e0.md](DERIVATION-08fcc6e0.md) · record
- [PREREGISTRATION-e0732f21.json](PREREGISTRATION-e0732f21.json) · result
- [STATIC_VERIFICATION_V1.json](STATIC_VERIFICATION_V1.json) · result
- [SOURCE_FREEZE_V1.json](SOURCE_FREEZE_V1.json) · result
- [K42_K51_SOURCE_ACTION_RANK0201_RUN.lock](K42_K51_SOURCE_ACTION_RANK0201_RUN.lock) · record

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-sixth-cumulant-4-2-5-1-pair-inheritance-residual) · [Copy provenance](../../../PROVENANCE.json)

