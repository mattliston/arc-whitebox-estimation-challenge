# Phase 2 role-target custody boundary

The official full Parquet layout cannot physically isolate observer roles. Its
63 row groups contain 8 or 16 networks each, every training row group is mixed,
and no target column/page index permits selective reads. Reading a raw
observer-fit row therefore also opens target pages containing audit or lockbox
networks.

The safe boundary is consequently two-stage:

1. A separately authorized trusted data custodian uses
   `authorized_full_targets.py` once. That interface projects only `mlp_id` and
   `all_layer_means`, truthfully opens all 1,000 targets, joins the frozen
   target-blind roster, and writes separate role artifacts. No ordinary
   training process may import or receive the raw full snapshot or this raw
   iterator.
2. The custodian places only `observer_fit.parquet`,
   `observer_selection.parquet`, and `manifest.json` in the training bundle.
   Audit, lockbox, and retired-diagnostic artifacts stay in separately
   controlled locations. `sealed_role_targets.py` rejects extra bundle files,
   checks every source/file/ordered-ID commitment, and physically projects only
   `mlp_id` plus `all_layer_means` after explicit authorization.

The training bundle manifest schema is
`whest-phase2-training-role-target-bundle-v1`. It must contain:

- `status`: `SEALED_BY_TRUSTED_DATA_CUSTODIAN`
- the frozen `split_manifest_sha256` and
  `role_addressability_audit_sha256`
- `target_columns`: `["mlp_id", "all_layer_means"]`
- `raw_full_dataset_included`: `false`
- `withheld_roles_included`: `false`
- exactly two `artifacts` entries: `observer_fit` and `observer_selection`
- for each artifact: a basename-only `path`, byte `sha256`, `rows`, and
  `ordered_mlp_ids_sha256`

`ordered_mlp_ids_sha256` is SHA-256 over each decimal `mlp_id` followed by a
newline, in ascending frozen `global_hash_rank`. Each role Parquet file must
contain exactly two columns in that order: int32 `mlp_id` and float32
`all_layer_means[16,1024]` carrying the Arrow `Array2D` extension metadata,
with rows in the same frozen order.

No custodian execution or role artifact exists yet. This protocol and both
readers are inert interfaces only; their preflights decode zero target values.
