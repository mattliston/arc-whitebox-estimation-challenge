# Linux science port

New files only. No VM commands, installs, full estimator imports or predictions were run while preparing this port. It has no fire command and is not a release gate. Original local release gates remain required.

The existing curated payload remains2,224,946,560bytes (2.072GiB). Add only this small source directory and its RUN_PLAN_v1.json to the transfer. Do not transfer __pycache__, incomplete diagnostic output, credentials, SSH keys, local virtualenv binaries or other store rows. The receiver must verify payload hashes; reference prediction symlinks must be dereferenced.

Three environments are needed: NumPy-free CPython3.10 client with flopscope-client0.12.3, server with flopscope/flopscope-server0.12.3+NumPy2.2.6+opt_einsum3.4.0, and NumPy judge. Exact currently installed package pins are in the original requirements drafts and run plan. This port checks those pins and every transferred SDK implementation/tariff fingerprint. If a Linux wheel differs, diagnose the mismatch; do not bypass billing/source verification. Python3.10 patch differences are recorded explicitly rather than silently claimed identical. No install is performed by these scripts.

The bank source is reconstructed from frozen d03b2500 plus the pinned final no-governor tail. Effective bank SHA256: af2023a220f24299bdf91e464ce809a51159b002e178a8f47dfad3fba7b4eb0a. Bridge source:51471b9b4f8408c6a651ad6d89641f9cb08df0055ca9285639df6236d7633813. No physics/precision/tariff changes. Both end in _FAST_RESCUE_FRAC={}. The bank reconstruction avoids transferring another2MiB duplicate.

The Linux server uses the stock SDK kernel implementation. Only array-store and communication marks are observed; Darwin cycle/instruction probes are omitted and unavailable fields are null. Official timing/billing remain untouched. Client fixture load,16layer views,predict,final fetch,finite checks and handle deletion are inside BudgetContext(2**41), as in the reference science worker. The passive product/prebind census is preserved. The original19000-request setup clock is imported unchanged under `--setup paced`; it is a science comparison condition, not a Linux setup-speed pass. `--setup real` is a separately labeled diagnostic.

## Receiving host commands

Use the repository-relative tree below an explicit bundle root. Substitute paths to the environments the lead provisions. These commands are prepared, not executed.

```sh
ARC_ROOT=/opt/arc-bundle
ARC_PORT=$ARC_ROOT/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/cloud_linux_v1
ARC_PLAN=$ARC_PORT/RUN_PLAN_v1.json
ARC_CLIENT=/opt/arc-env/client/bin/python
ARC_SERVER=/opt/arc-env/server/bin/python
ARC_JUDGE=/opt/arc-env/judge/bin/python
ARC_STATE=/opt/arc-runs/admission
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1

# This verifies source pins plus all32weights/64truths and makes a new receipt-plan.
$ARC_JUDGE "$ARC_PORT/prepare_plan_v1.py" --root "$ARC_ROOT" --transfer-manifest "$ARC_ROOT/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_vm_v1/TRANSFER_MANIFEST_v1.json" --verify-data --out /opt/arc-runs/VERIFIED_PLAN_v1.json
ARC_PLAN=/opt/arc-runs/VERIFIED_PLAN_v1.json
$ARC_CLIENT "$ARC_PORT/inspect_environment_v1.py" --root "$ARC_ROOT" --plan "$ARC_PLAN" --role client --out /opt/arc-runs/client_environment.json
$ARC_SERVER "$ARC_PORT/inspect_environment_v1.py" --root "$ARC_ROOT" --plan "$ARC_PLAN" --role server --out /opt/arc-runs/server_environment.json
$ARC_JUDGE "$ARC_PORT/smoke_guard_linux_v1.py" --out /opt/arc-runs/guard_smoke --state-dir "$ARC_STATE"
```

Create `/opt/arc-runs` with the intended worker owner first. All output paths must be new. Optional `--cgroup-root /sys/fs/cgroup/arcworker` requires an already delegated cgroup-v2 parent with memory controller enabled. The guard fails if a requested cgroup cannot be configured. It never creates delegation or changes system-wide limits. With delegation it sets memory.max7,900,000,000, memory.swap.max0 and memory.oom.group1 before releasing the gated child. Without delegation it reports a sampled process-tree RSS fallback; this is not a kernel hard cap or macOS footprint. Both modes kill the whole owned group on cap, swap, timeout or loss of the reserved available memory. Atomic admission reserves the unconsumed part of every admitted7.9GB budget across concurrent controllers. Start with the tiny guard smoke, then the quiet numerical bridge.

## Quiet bridge, then scaling

Run group1=[35,150,200] in order bank1,bridge1,bridge2,bank2 with `--jobs1`, distinct output directories and no other VM work. Example for one of the four sessions:

```sh
$ARC_JUDGE "$ARC_PORT/run_science_linux_v1.py" --root "$ARC_ROOT" --plan "$ARC_PLAN" --candidate bank337393 --groups 1 --jobs 1 --out /opt/arc-runs/bank1 --client-python "$ARC_CLIENT" --server-python "$ARC_SERVER" --state-dir "$ARC_STATE"
# Repeat with bridge_micro/outbridge1, bridge_micro/outbridge2, bank337393/outbank2.
$ARC_JUDGE "$ARC_PORT/grade_linux_v1.py" --root "$ARC_ROOT" --plan "$ARC_PLAN" --candidate /opt/arc-runs/bridge1 --bank /opt/arc-runs/bank1 --ranks 35,150,200 --out /opt/arc-runs/bridge_pilot_grade.json
```

Before scaling, compare repeated same-source prediction hashes,bills,requests,actual setup coverage,prebind keys and product_geometry_counts. Compare against the transferred Mac sources' same original group rows, not old95c controls. Investigate any repeated same-source nondeterminism or bill/key discrepancy. Cross-architecture output bytes may differ; score candidate-versus-bank on Linux, and compare that contrast to the Mac-versus-Mac contrast. Do not divide Linux candidate U by Mac bank U. No cloud residual timing claim transfers to the public grader.

After the bridge is understood, start2lanes then4 and expand. Target an initial calibrated ceiling of12lanes on the stated180GB VM,8on the stated120GB VM; fewer if measured throughput or peaks warrant it. Every group is still7.9GB with global available reserve max(16GB,20%effective RAM). These are capacity ceilings, not permission to skip calibration. `--jobs` is also the shared admission ceiling. To run all32 omit `--groups`, keep each original group intact, and grade all32 with first7/held25 separately. A failed group stops new admission and retires the controller's active owned groups; incomplete results cannot be graded. Original files are never overwritten.

## Fresh fixtures

The minimal transfer intentionally has no fresh weights/truths or arbitrary loader cache. `prepare_fresh_plan_v1.py` consumes a separate frozen manifest of only the selected fresh weight.npy and truthnpz files with explicit SHA256/size/key. It neither generates truth nor imports a network. It makes singleton seed groups. Run both bank and candidate with `--warmup-each` on the same plan; the scorer requires matching warm mode. Production-boundary fixture load remains metered, so this is a paired Linux fresh protocol rather than a claim of identical absolute bills to the local in-processv9 evaluator. Use new fixed seeds, not held25 fitting. No extra fresh data is in the current2.072GiB payload.

## Local validation and remaining acceptance

All Python files compile and the frozen run plan verifies its source pins locally. The offline scorer is checked against the saved32Mac bank/bridge outputs and64truth files. No Linux worker, guard or estimator has run yet. The VM must pass environment/source verification, the tiny guard cleanup smoke, then the quiet numerical bridge. Hardware-counter and old GC diagnostic timing fields are intentionally unavailable; they are not fabricated zero readings. The observer wrappers may change Python overhead, so this port only supplies scientific accuracy/billing bandwidth until locally requalified.
