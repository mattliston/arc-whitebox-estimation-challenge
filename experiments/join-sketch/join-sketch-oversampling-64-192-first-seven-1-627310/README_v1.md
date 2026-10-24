# Fresh gate re-adaptation screen

Prepared at 2026-10-20T21:03:07Z; no controller launched by the joint agent. Frozen source, dependency and truth pins are in `MANIFEST_v1.json`. The exact six settings are in `settings_v1.json`; every setting states both coefficients and clears both `_GQ33_CONST` and `_GSK_CONST`.

Candidate: `candidate_micro_copyall_ecut_nogov_astra_sci_v1.py`, SHA `51aec0294ab23ced8280cfd104d6ebc97647585904721822c4b96be3049af6d5`. Paired bank: frozen #337393 release `fd8739582ffdd04decc5fe6bbb2f99fb0b3dc02a2ee786d684bd21bbf52d974d`, using `[{}]` and the exact existing v9 warm-up. The new candidate's last source statement is `_FAST_RESCUE_FRAC = {}`. The frozen bank keeps its original code and is evaluated under v9's existing science rescue cap `1e9`.

`check_cache_ast_v1.py` extracts only the two `_gq` definitions. The tiny NumPy proof demonstrates the stale-cache failure without resets, then real live/legacy coefficient changes across all six settings, same-width reuse, and return to exact defaults. It does not import or set up an estimator. At width1024 the reset rebuilds three constant arrays containing1159 elements; actual bill includes them. The fixed polynomial has unchanged operation shapes, but altered network states can still change later routes and total bill.

The four truth fixtures are present, finite and seed-matched, with64,094,208 samples each (two32,047,104 halves). Their stored weight digests were recorded; preparation did not regenerate full networks.

From the repository root, the lead can launch these independently after checking capacity:

```bash
.venv/bin/python -B pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v1.py --mode primary --run
.venv/bin/python -B pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v1.py --mode aux --run
```

Without `--run`, the controller only checks pins and prints its partition. The primary uses the lead-created `queue_v14.txt.reserve_gate_coadapt` and seeds7100092–93; the auxiliary uses the existing `AUXILIARY32_SLOT_v1.lock`, requires at least60% free before each role shard, and uses seeds7100094–95. Bank then screen run sequentially for each seed. Both preserve the existing native7.9GB footprint /9.5GB RSS policy; primary hard-free floor15%, auxiliary30%. Each guard gets600seconds. No existing runner/guard is edited.

There are36 total predictions including warm-ups and28 saved scored predictions. A failed/retired shard stops its partition and preserves files. `--resume` reuses only fully verified complete role shards; it refuses partial output. SIGTERM/INT requests stopping after the current guarded shard retires. The primary reservation is removed when its controller exits normally or after a guarded failure; the auxiliary lock is process-owned. Inspect guard receipts before any manual retry.

`analyze_v1.py` runs automatically only when all four paired seed results exist. It verifies prediction hashes, guard retirement and truth-derived U, then writes `RESULT_FRESH4_v1.json`. Every reported gain is the mean of per-seed actual `U×bill` ratios against the matching bank337393 prediction; the screen-default contrast is separately labelled. No old average is added. This is a six-arm fresh screen, not current32 qualification, a new release wrapper, or permission to fire.
