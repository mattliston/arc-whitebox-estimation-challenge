# S47 pair qualification — Astra release lane

Prepared 2026-10-10; no session is armed by this document. The accepted trainer
handover, untouched-network judgment and native parity are prerequisites. The
engineer expects new learned arrays with the S46 forms, hence no new einsums.

1. Record the engineer's exact safe source SHA, build spec, head-table SHA,
   per-layer parity on 651/378, and ten paired native networks against S46.
   Record the physicist's untouched seed assignment and verdict.
2. Build safe from that exact source, in a NEW variant directory:
   `python -B build_variant.py <engineer-safe.py> <safe-name> l3_bind128 lvfs_limit_off prices_zero`.
3. Make a versioned builder copy (archive the prior copy before any refresh).
   First reproduce the engineer's SHA byte for byte. Prefer his qualified twin
   trade set when a gain is small; the aggressive release twin is optional.
   For the aggressive twin: no L3/L3o, S42 fusion, all planner prices zero, and a
   base with no price override appended after Rprices_zero. K4Lv7/F4Cv3 require
   DLY fragment v2. The safe source and twin must use the same learned physics.
4. Stage A each source against the LAST FIRED source, not just the older bank.
   Initially pass no `--reviewed`; inspect every new einsum before an explicitly
   reviewed rerun under a new tag. Deep-layer runtime safety is stage C's job.
5. Root coordinates load before C. One release session at a time, nice 10,
   OMP/OpenBLAS/MKL threads 1. Reserve about 30 minutes for four sessions on a
   quiet machine (longer under load). No concurrent native engineer sessions or
   offline fits, and do not stop/restart existing trainer work without root's
   coordination. Keep the built-in own_gate/footprint/swap guards enabled.
6. Run safe C and slow setup, judge, then the strict pre-fire audit below; repeat
   for twin with `--primary` on designation_judge. Use new tags and exclusive
   output files on EVERY attempt. A failed/contaminated run is preserved and
   remeasured under another tag. Never take a failed process exit as success.
7. Send root the resulting hashes, gate report and numeric judge rows. The
   strict checker is release-only when stage D is skipped. A complete final
   designation qualification still needs stage D's worker memory/handles,
   finite outputs and native bytes/bill checks. Do not describe skipped D as
   passed. Root alone fires, polls and manages the daily submission cap.

Run the following from `O/release`, with shell variables assigned to concrete
NEW names after the handover. `$PY` is the workspace `.venv/bin/python`.
`$SOURCE` is the exact candidate; `$TAG` is its session prefix;
`$ATAG` is its already packaged stage-A directory; `$MODE` is empty for safe or
`--primary` for twin. These commands must be sequential and each must succeed.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
  "$PY" -B stage_c_boundary.py "$SOURCE" "c_$TAG" --short-setup --legs C \
  --parent-from runs/c_S42_bank_legs_heavy
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
  "$PY" -B stage_c_boundary.py "$SOURCE" "${TAG}_slow" --setup-slowdown 1.3 --legs C \
  --parent-from runs/c_S42_bank_legs_heavy
set -o noclobber
"$PY" -B designation_judge.py "$TAG" --pair "runs/c_$TAG" \
  --slow "runs/${TAG}_slow" --worker runs/none $MODE > "runs/judge_${TAG}.txt"
"$PY" -B release_gate_v1.py "$TAG" --stage-a "runs/$ATAG" \
  --pair "runs/c_$TAG" --slow "runs/${TAG}_slow" \
  --judge "runs/judge_${TAG}.txt" --allow-missing-worker > "runs/gate_${TAG}.json"
```

The judge prints its verdict but does not fail its process on a failed numeric
row. `release_gate_v1.py` supplies the exit status: it checks every raw A/C/slow
gate, exact source hashes and archive membership, plus the judge's exact verdict.
Its allowed missing-worker verdict rejects all additional missing or spoiled
evidence. The previous chain's grep could accept a paged pair with the same
missing-worker exception; none of S46's actual raw sessions has that problem.

S46 remains the graded pair: safe #335259 and engineer twin #335261, both 100/100
networks, 1.581e-9 public score. Backup `af0cd8e4` has only stage A; it has no C,
slow-setup, or worker qualification. Its estimated -0.17% bill and roughly 10.5%
loss risk are historical extrapolations, not a measured release qualification.
