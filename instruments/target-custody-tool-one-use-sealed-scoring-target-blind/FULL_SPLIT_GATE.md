# Phase 2 full-split data gate

Status: the pinned 1,000-network full split is cached, byte-verified, and ready
for target-blind weight passes. The deterministic network roster is frozen and
reproducible. No target values were decoded while building or auditing this
gate, and no target role artifacts have been created.

## Pinned source

- repository: `aicrowd/arc-whestbench-public-2026`
- tag: `v2-phase2`
- immutable commit: `ce926b7d1ced65faa668c01b94a43f8c4f43783d`
- Hub config: `full`
- exact objects: 63 Parquet shards, 1,000 networks, 69,955,020,892 bytes
  (65.15069016441703 GiB)
- canonical exact-shard manifest SHA-256:
  `51357aceca60f0a0365b26d7b25e136d1e4e713d44451a5f7c079e3ba9ea2794`

The official sidecar names the config `default`, while the Hub dataset server
exposes it as `full`. The downloader therefore bypasses wrapper/config
resolution and requests the exact 63 authority-listed paths at the immutable
commit.

Exact download/replay command:

```sh
/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/bin/python /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/download_full_split.py --execute --max-workers 4
```

The command has been replayed against the complete cache and returned
`PASS_DOWNLOAD_EXACT_63_SHARDS_POINTER_SIZE_AND_SCHEMA`.

## Target-blind access

`full_data_gate.py` physically projects only `mlp_id`, `mlp_name`, `mlp_seed`,
and `weights`. The other four source columns are deny-listed. Every cached
object was independently SHA-256 checked as opaque bytes, then its footer,
schema, geometry, and row count were verified. Mini-cache and Phase 1 artifacts
remain untouched.

## Frozen roster

Assignment is exactly:

`SHA256("p2-teacher-v1|v2-phase2|" + signed_decimal_mlp_seed)`, sorted by digest.

Duplicate seeds or assignment digests fail. `mlp_id` and the SHA-256 of each
float32 `[16,1024,1024]` weight tensor are unique identity attestations only;
they do not affect assignment.

- primary splits: 700 train, 100 calibration, 100 validation, 100 test
- ranks 0:12: retired teacher diagnostics
- ranks 12:700: observer fit (688)
- ranks 700:712: retired teacher diagnostics
- ranks 712:800: observer selection (88)
- ranks 800:900: observer audit (100)
- ranks 900:1000: observer lockbox (100)

The complete 1,000-network roster was rebuilt from the four-column target-blind
projection and matched `full_network_splits.json` byte for byte.

## Honest target boundary

Raw role-scoped target reads are not physically safe. The upstream layout has
63 row groups (62 of 16 rows and one of 8), zero single-network row groups,
zero role-pure row groups, and no target column/page indexes. Training roles
share 62 row groups with audit or lockbox networks.

`authorized_full_targets.py` is therefore restricted to a separately
authorized trusted custodian whose authorization covers the complete raw
`all_layer_means` column. Ordinary training raw access is forbidden.
`sealed_role_targets.py` is the later training interface: it accepts only a
hash-bound bundle containing the fit and selection role artifacts and rejects
extra files, audit/lockbox roles, weights, and auxiliary columns. See
`ROLE_TARGET_CUSTODY_PROTOCOL.md`.

Both interfaces are currently inert. Their preflights decode zero targets.

## Frozen receipts

- parent authority SHA-256:
  `b33aea75ee83be5cc7c2778aefd4cc35d731dcd702d8b258269fc2c895ef4ec5`
- full authority extension SHA-256:
  `d61677658462f22949d8242f7bb92b24341502678f499d793184002fd3b6027a`
- remote authority audit SHA-256:
  `a5f9e0b42b14f36a2736555d1c9a4a018d8cc5e6935937b40615d79d37edda10`
- full cache byte/schema audit SHA-256:
  `43857f77f86f2ce95f058bda7d314d7c836b60aedab5a40d37b7208f291494fd`
- frozen-prefix preservation audit SHA-256:
  `6569d478615c1a68818229bf328b93340f342296be14cd5529b220e007961747`
- target-blind roster SHA-256:
  `ba325d5c9ee4382ebf856b2e8d15330852fba5ac5627665fcd81dd9a5ef461d1`
- raw role-addressability audit SHA-256:
  `b73e6ccf24434d27cc1bc34cc5ebcd87b9b38b504c3412bf9afad571ecb4b49c`
- gate structure audit SHA-256:
  `8e75e15b52d411be194becf5f2f956e09b6b62d7970b9d0ea2d47a21e4cd5b77`
