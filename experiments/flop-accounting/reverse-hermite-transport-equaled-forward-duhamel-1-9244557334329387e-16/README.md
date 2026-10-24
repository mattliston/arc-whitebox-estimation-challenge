# Reverse-Hermite transport equaled forward Duhamel to 1.9244557334329387e-16

2026-08-26 · experiment · rejected · measured

Closed at the exact tested scope. RESULT.json (SHA-256 785d789a9ce588989f6ab070d84c6dfdf8ec4faaacc3eabf1bbe791fb49b70e6) proves the compressed reverse transport and forward Duhamel sum for fixed sources agree to 1.9244557334329387e-16, so reversing direction adds no new operator at the same finite truncation. The target-free K5/K7/K9/K12 source ladder fails the frozen accuracy, alignment, degree, stability, finite-width, and novelty gates: K12 geometric-mean baseline/corrected mean squared error (MSE) improvement is 0.000131045125647072 (required >=3 on every network), and even K5's best single-network improvement is only 1.4682954929865684.

Recorded measurement:

    Closed at the exact tested scope. RESULT.json (SHA-256 785d789a9ce588989f6ab070d84c6dfdf8ec4faaacc3eabf1bbe791fb49b70e6) proves the compressed reverse transport and forward Duhamel sum for fixed sources agree to 1.9244557334329387e-16, so reversing direction adds no new operator at the same finite truncation. The target-free K5/K7/K9/K12 source ladder fails the frozen accuracy, alignment, degree, stability, finite-width, and novelty gates: K12 geometric-mean baseline/corrected MSE improvement is 0.000131045125647072 (required >=3 on every network), and even K5's best single-network improvement is only 1.4682954929865684. The literal non-equivalent degree-5 scalar basis needs at least 9,474,438,804,480 homogeneous coefficients, already 4.31x the complete 2^41 FLOP budget before transport or evaluation. This closes the cheap finite reverse-Hermite spelling and literal full-basis alternative only; B1B preserves a future named structured or factorized representation.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b1a_reverse_hermite_independent_audit_v1/verify_audit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b1a_reverse_hermite_independent_audit_v1/verify_audit.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b1a_reverse_hermite_v1/run_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b1a_reverse_hermite_v1/run_discriminator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_exact_ledger_v1/ledger_rig.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_exact_ledger_v1/ledger_rig.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_exact_ledger_v1/ledger_verify.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_exact_ledger_v1/ledger_verify.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_t34_v1/t34_supply_core.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_t34_v1/t34_supply_core.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_missing_routes_v1/missing_routes_core.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_missing_routes_v1/missing_routes_core.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_supply_card_v1/build_estimator_supply.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_supply_card_v1/build_estimator_supply.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_supply_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_supply_card_v1/estimator.py)

## Evidence

- [PREREGISTRATION.json#L1](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/PREREGISTRATION.json#L1)
- [PREREGISTRATION_SEAL.json#L1](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/PREREGISTRATION_SEAL.json#L1)
- [RESULT.json#L917](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/RESULT.json#L917)
- [TERMINAL_RECEIPT.json#L21](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/TERMINAL_RECEIPT.json#L21)
- [AUDIT_RECEIPT.json#L43](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/AUDIT_RECEIPT.json#L43)
- [INDEPENDENT_AUDIT.md#L334](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/INDEPENDENT_AUDIT.md#L334)

<details>
<summary>All result files and records (6)</summary>

- [PREREGISTRATION.json](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/PREREGISTRATION.json) · result
- [PREREGISTRATION_SEAL.json](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/PREREGISTRATION_SEAL.json) · result
- [RESULT.json](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/RESULT.json) · result
- [TERMINAL_RECEIPT.json](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/TERMINAL_RECEIPT.json) · result
- [AUDIT_RECEIPT.json](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/AUDIT_RECEIPT.json) · result
- [INDEPENDENT_AUDIT.md](../reverse-hermite-discriminator-direction-reversal-added-no-operator-fixed/INDEPENDENT_AUDIT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-reverse-hermite-transport-equaled-forward-duhamel-1-9244557334329387e-16) · [Copy provenance](../../../PROVENANCE.json)

