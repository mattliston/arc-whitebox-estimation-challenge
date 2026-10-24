# FlopScope 0.12 retained diagnostic operation histories across rows and exhausted memory

2026-08-28 · incident · failed · measured

Runtime V4 made exact claim d690aa5b and was interrupted after confirming near-linear retained diagnostic history. Failure fbac37a9 is terminal post-claim/no-retry with no published bundle. Independent incident audit 060e7bab identifies the FlopScope 0.12 process-global BudgetAccumulator retaining each completed row's diagnostic OpRecord list; it reports zero bank payload, row-manifest, weight, target, fit, score, or submission access during the audit.

Recorded measurement:

    Runtime V4 made exact claim d690aa5b and was interrupted after confirming near-linear retained diagnostic history. Failure fbac37a9 is terminal post-claim/no-retry with no published bundle. Independent incident audit 060e7bab identifies the FlopScope 0.12 process-global BudgetAccumulator retaining each completed row's diagnostic OpRecord list; it reports zero bank payload, row-manifest, weight, target, fit, score, or submission access during the audit. This closes V4 only.

## Programs

- [../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/capture_adapter.py](../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/capture_adapter.py)
- [../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/parent_capture.py](../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/parent_capture.py)
- [runtime_runner.py](runtime_runner.py)
- [../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/runtime_token.py](../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/runtime_token.py)
- [../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/signed_pair_literal.py](../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/signed_pair_literal.py)
- [../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/signed_pair_reference.py](../2026-08-27-signed-pair-runtime-rejected-allowed-audit-counter-before-generating/signed_pair_reference.py)
- [capture_adapter.py](capture_adapter.py)
- [parent_capture.py](parent_capture.py)
- [runtime_token.py](runtime_token.py)
- [signed_pair_literal.py](signed_pair_literal.py)
- [signed_pair_reference.py](signed_pair_reference.py)

## Evidence

- [INDEPENDENT_STATIC_REVIEW_V4.json#L1](INDEPENDENT_STATIC_REVIEW_V4.json#L1)
- [THREE_BANK_RUN.CLAIM.json#L1](THREE_BANK_RUN.CLAIM.json#L1)
- [THREE_BANK_RUN.FAILURE.json#L1](THREE_BANK_RUN.FAILURE.json#L1)
- [INDEPENDENT_INCIDENT_ROOT_CAUSE_V1.json#L1](INDEPENDENT_INCIDENT_ROOT_CAUSE_V1.json#L1)
- [V4_TERMINAL_FAILURE_LINEAGE_V5.json#L1](V4_TERMINAL_FAILURE_LINEAGE_V5.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [INDEPENDENT_STATIC_REVIEW_V4.json](INDEPENDENT_STATIC_REVIEW_V4.json) · result
- [THREE_BANK_RUN.CLAIM.json](THREE_BANK_RUN.CLAIM.json) · result
- [THREE_BANK_RUN.FAILURE.json](THREE_BANK_RUN.FAILURE.json) · result
- [INDEPENDENT_INCIDENT_ROOT_CAUSE_V1.json](INDEPENDENT_INCIDENT_ROOT_CAUSE_V1.json) · result
- [V4_TERMINAL_FAILURE_LINEAGE_V5.json](V4_TERMINAL_FAILURE_LINEAGE_V5.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-28-flopscope-0-12-retained-diagnostic-operation-histories-across-rows) · [Copy provenance](../../../PROVENANCE.json)

