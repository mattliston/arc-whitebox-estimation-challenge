# Target-custody tool: one-use sealed scoring, target-blind loaders and role-specific target bundles

2026-08-23T17:22Z · tool · adopted · derived

The data gate separated prediction from scoring: estimators ran on a target-blind loader that decodes only the weight columns of the v2-phase2 dataset, predictions were sealed with hashes, and the all-layer means of the mini split were opened once per scoring release under a durable one-use claim recorded in a receipt. A production custodian later sealed role-specific target bundles (fit, selection, audit and lockbox positions over the 1000 full networks) so that observers could be fitted without touching public leaderboard names. Every score in the first days of Phase 2 carries one of these claim and release receipts.

Recorded measurement:

    mini split 100 networks, full split 1000; the V2 full-budget joint scoring opened 100 target rows (1,638,400 float32 values) exactly once; fit/selection/audit/lockbox positions 12-700 / 712-800 / 800-900 / 900-1000

## Programs

- [code/target_blind_mini.py](code/target_blind_mini.py)
- [code/custodian_seal_training_roles_once_v3.py](code/custodian_seal_training_roles_once_v3.py)
- [code/audit_role_target_addressability.py](code/audit_role_target_addressability.py)
- [code/authorized_full_targets.py](code/authorized_full_targets.py)
- [code/authorized_full_targets_v2.py](code/authorized_full_targets_v2.py)
- [code/build_full_network_splits.py](code/build_full_network_splits.py)
- [code/custodian_seal_training_roles_once.py](code/custodian_seal_training_roles_once.py)
- [code/custodian_seal_training_roles_once_v2.py](code/custodian_seal_training_roles_once_v2.py)
- [code/download_full_split.py](code/download_full_split.py)
- [code/full_data_gate.py](code/full_data_gate.py)
- [code/role_target_artifact_codec_v1.py](code/role_target_artifact_codec_v1.py)
- [code/run_synthetic_custodian_chain_v3.py](code/run_synthetic_custodian_chain_v3.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/sealed_role_targets.py](code/sealed_role_targets.py)
- [code/sealed_role_targets_v2.py](code/sealed_role_targets_v2.py)
- [code/target_blind_mini_v2.py](code/target_blind_mini_v2.py)
- [code/verify_custodian_compat_chain_v3.py](code/verify_custodian_compat_chain_v3.py)
- [code/verify_custodian_sealer_static.py](code/verify_custodian_sealer_static.py)
- [code/verify_custodian_sealer_static_v2.py](code/verify_custodian_sealer_static_v2.py)
- [code/verify_full_network_splits.py](code/verify_full_network_splits.py)

</details>

## Evidence

- [fullbudget-joint-mini100-v2.joint-raw-mse.json#L1](fullbudget-joint-mini100-v2.joint-raw-mse.json#L1)
- [BUDGET_AUDIT.json#L1](BUDGET_AUDIT.json#L1)

<details>
<summary>All result files and records (7)</summary>

- [ROLE_TARGET_CUSTODY_PROTOCOL.md](ROLE_TARGET_CUSTODY_PROTOCOL.md) · record
- [FULL_SPLIT_GATE.md](FULL_SPLIT_GATE.md) · record
- [authority.json](authority.json) · config
- [full_network_splits.json](full_network_splits.json) · config
- [PRODUCTION_CUSTODIAN_V3_ONE_USE_AUTHORIZATION.json](PRODUCTION_CUSTODIAN_V3_ONE_USE_AUTHORIZATION.json) · record
- [fullbudget-joint-mini100-v2.joint-raw-mse.json](fullbudget-joint-mini100-v2.joint-raw-mse.json) · record
- [BUDGET_AUDIT.json](BUDGET_AUDIT.json) · record

</details>

[Research graph](../../graph/index.html#node=2026-08-23-target-custody-tool-one-use-sealed-scoring-target-blind) · [Copy provenance](../../PROVENANCE.json)

