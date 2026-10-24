#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-DS = the engineer's E-46 depth_shadow_v6.py + SQUARES: --kinds=sq executes every square product routed
through _mm (max dim <= 1024) at its native depth AND at six levels (SQUARE6 layout) in place, random order, only the native output
goes on; squares under the W5 port grant and natives below 4 are skipped; batched products (_mm_batch) are untouched. Outputs
land in fable_engineer2_v1/depth6/. Original docstring follows.) (E-46 v6: v5 + --mlock=1, which wires the kernel arena after the first net, so the memory compressor cannot
page it out between uses; tests whether the six-level excess is paging of idle arena regions. v5: user and system CPU and page
faults per execution): IN-SITU PAIRED depth pricing on flopscope backend time (engineer2's local analog of the platform
kernel wall). Whole-net predictions run in process (OpenBLAS env, 1 BLAS thread, nets back to back, first net cold). Every Winograd
push (and, with --kinds=push,read, every read) is executed at EVERY allowed depth right where the network calls it, in a random
order per call, so all depths of one call see the same operands, the same load and the same cache history; only the output of the
KEPT depth goes on (the others are discarded), so the network itself runs exactly the kept policy.
  allowed push depths: 1024-row pushes 4, 5, 6 (6 = the zero-padded six-level push when k or n is not a multiple of 64, as push5p6
  does); other pushes with k <= 1024: 4, 5. Reads: 4, 5. A product whose native rule says 0 (plain matmul) is left alone.
  --keep=native (the candidate's own rule) | s6b (the push5 rule of S6b: no padded six) | table:<file.json> ({"kind,m,k,n": depth};
  unlisted shapes native) | force:<L> (every push at L where allowed, else the nearest lower allowed depth)
  --shadow=1 (run all depths per call; default) | 0 (run only the kept depth: a plain policy run, for whole-net pairs)
Per net: totals (bill, backend, process CPU, wall), the kept-only totals (totals minus the discarded executions), the kept
prediction (npy, for bit-exactness), and per (kind, m, k, n, depth): calls, backend sum / sum of squares, bill, and backend by
execution position (first/second/third), to expose order effects.
usage: depth_shadow_v6.py <candidate.py> <ranks> <tag> [--keep=...] [--shadow=1] [--kinds=push] [--seed=0]
  -> fable_engineer_v1/depth_shadow_<tag>.json (v2: same outputs; cells gain counted ops; classes = [shape, native, kept]) (+ _pred_r<rank>.npy)"""
import sys, json, types, time, random, collections, resource, ctypes
from pathlib import Path
import numpy as np
import flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--")); pos = [a for a in sys.argv[1:] if not a.startswith("--")]
cand = Path(pos[0]).resolve(); ranks = [int(x) for x in pos[1].split(",")]; tag = pos[2]
KEEP = opts.get("keep", "native"); SHADOW = int(opts.get("shadow", "1")); KINDS = set(opts.get("kinds", "push").split(","))
rng = random.Random(int(opts.get("seed", "0")))
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("depth_shadow_e2_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
assert hasattr(mod, "_dr_value") and hasattr(mod, "_DR_BANK_RULE"), "candidate lacks the depth-rule fragment"
live_rule = mod._rule_depth; orig_mm = mod._mm
TABLE = {}
if KEEP.startswith("table:"):
    TABLE = {tuple([k.split(",")[0]] + [int(x) for x in k.split(",")[1:]]): int(v) for k, v in json.loads(Path(KEEP[6:]).read_text()).items()}

def allowed(kind, m, k, n):
    if kind == "sq":
        return (4, 5, 6)
    if kind == "read":
        return (4, 5)
    if m <= 1024 and k <= 1024:
        return (4, 5, 6) if m == 1024 else (4, 5)
    return None

def s6b_rule(kind, m, k, n, native):
    if kind != "push" or not (m <= 1024 and k <= 1024):
        return native
    options = (4, 5, 6) if (m == 1024 and m % 64 == k % 64 == n % 64 == 0) else (4, 5)
    return min(options, key=lambda L: mod._dr_value("push", m, k, n, L, mod._DR_MU))

def kept_depth(kind, m, k, n, native, options):
    if KEEP == "native":
        return native
    if KEEP == "s6b":
        return s6b_rule(kind, m, k, n, native)
    if KEEP.startswith("table:"):
        return TABLE.get((kind, m, k, n), native)
    if KEEP.startswith("force:"):
        L = int(KEEP[6:])
        return max(d for d in options if d <= L) if kind == "push" else native
    raise SystemExit(f"unknown --keep {KEEP}")

force = [None]
def rule_wrap(kind, m, k, n, maximum=6):
    f = force[0]
    if f is not None and kind == f[0]:
        return f[1]
    return live_rule(kind, m, k, n, maximum)
mod._rule_depth = rule_wrap

ctx_ref = [None]
acc = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0])         # (kind, m, k, n, depth) -> [calls, backend, backend^2, bill, counted ops]
kept_of = {}                                                        # (kind, m, k, n) -> (native depth, kept depth)
call_log = []                                                       # [call id, kind, m, k, n, depth, position, backend, bill, ops, cpu, user cpu, system cpu, minor faults, major faults]
by_pos = collections.defaultdict(lambda: [0, 0.0])                  # (kind, m, k, n, depth, position) -> [calls, backend]
shadow_tot = [0.0, 0]                                               # discarded executions: backend, bill

def classify(a, b):
    if not mod._WINOGRAD or str(a.dtype) != "float32" or str(b.dtype) != "float32" or len(a.shape) != 2 or len(b.shape) != 2:
        return None
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    if min(m, k, n) == 0 or n > mod._PACKED_TILE or k > mod._PACKED_TILE or m % 32 or k % 32 or n % 32:
        return None
    if max(m, k, n) <= 1024:
        if "sq" not in KINDS:
            return None
        scope = getattr(mod, "_W5_PORT_SCOPE", None)
        if scope is not None and scope.get("grant") is not None:
            return None
        native = mod._square_schedule(m, k, n)[0]
        if native < 4:
            return None
        options = tuple(sorted({native, 6}))
        return "sq", m, k, n, native, options
    kind = "push" if n > 1024 else "read"
    if kind not in KINDS:
        return None
    native = live_rule(kind, m, k, n)
    options = allowed(kind, m, k, n)
    if native == 0 or options is None or native not in options:
        return None
    return kind, m, k, n, native, options

def mm_wrap(a, b):
    c = ctx_ref[0]
    cl = classify(a, b) if c is not None else None
    if cl is None:
        return orig_mm(a, b)
    kind, m, k, n, native, options = cl
    keep = kept_depth(kind, m, k, n, native, options)
    kept_of[(kind, m, k, n)] = (native, keep)
    order = list(options) if SHADOW else [keep]
    if SHADOW:
        rng.shuffle(order)
    result = None; cid = len(call_log)
    for p, d in enumerate(order):
        t0, f0, q0, u0 = c.flopscope_backend_time_s, c._flops_used, len(c._op_log), time.process_time(); ru0 = resource.getrusage(resource.RUSAGE_SELF)
        force[0] = (kind, d)
        try:
            r = orig_mm(a, b)
        finally:
            force[0] = None
        dt, df, dq, du = c.flopscope_backend_time_s - t0, c._flops_used - f0, len(c._op_log) - q0, time.process_time() - u0; ru1 = resource.getrusage(resource.RUSAGE_SELF)
        v = acc[(kind, m, k, n, d)]; v[0] += 1; v[1] += dt; v[2] += dt * dt; v[3] += df; v[4] += dq
        w = by_pos[(kind, m, k, n, d, p)]; w[0] += 1; w[1] += dt
        call_log.append([cid, kind, m, k, n, d, p, round(dt, 6), int(df), int(dq), round(du, 6), round(ru1.ru_utime - ru0.ru_utime, 6),
                         round(ru1.ru_stime - ru0.ru_stime, 6), ru1.ru_minflt - ru0.ru_minflt, ru1.ru_majflt - ru0.ru_majflt])
        if d == keep:
            result = r
        else:
            shadow_tot[0] += dt; shadow_tot[1] += df
            del r
    return result
mod._mm = mm_wrap

man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    acc.clear(); by_pos.clear(); kept_of.clear(); call_log.clear(); shadow_tot[:] = [0.0, 0]
    t0 = time.time(); c0 = time.process_time(); rn0 = resource.getrusage(resource.RUSAGE_SELF)
    with flops.BudgetContext(flop_budget=2**44, quiet=True) as ctx:
        ctx_ref[0] = ctx
        pred = est.predict(MLP([W[l] for l in range(16)], f"ds-{rank}"), 2**41)
        ctx_ref[0] = None
    bill, backend = int(ctx._flops_used), float(ctx.flopscope_backend_time_s)
    np.save(HERE / f"depth_shadow_{tag}_pred_r{rank}.npy", np.asarray(pred)); ops = len(ctx._op_log)
    rn1 = resource.getrusage(resource.RUSAGE_SELF)
    rec = dict(rank=rank, keep=KEEP, shadow=SHADOW, bill=bill, backend=round(backend, 4), cpu=round(time.process_time() - c0, 2), wall=round(time.time() - t0, 2),
               utime=round(rn1.ru_utime - rn0.ru_utime, 2), stime=round(rn1.ru_stime - rn0.ru_stime, 2), minflt=rn1.ru_minflt - rn0.ru_minflt, majflt=rn1.ru_majflt - rn0.ru_majflt,
               kept_bill=bill - shadow_tot[1], kept_backend=round(backend - shadow_tot[0], 4), ops=ops,
               cells=[[list(k), v[0], round(v[1], 5), round(v[2], 7), int(v[3]), int(v[4])] for k, v in sorted(acc.items())],
               classes=[[list(k), v[0], v[1]] for k, v in sorted(kept_of.items())], calls=list(call_log),
               positions=[[list(k), v[0], round(v[1], 5)] for k, v in sorted(by_pos.items())])
    out.append(rec)
    if int(opts.get("mlock", "0")) and not getattr(mod, "_E46_LOCKED", False):
        arena = mod._SCRATCH[0]; libc = ctypes.CDLL(None, use_errno=True); libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        rc = libc.mlock(ctypes.c_void_p(arena.ctypes.data), ctypes.c_size_t(arena.nbytes)); mod._E46_LOCKED = True
        print(json.dumps(dict(mlock_rc=rc, errno=ctypes.get_errno(), arena_mb=round(arena.nbytes / 2**20))), flush=True)
    push = [x for x in rec["cells"] if x[0][0] == "push"]
    print(json.dumps(dict(rank=rank, kept_util=round(rec["kept_bill"] / 2**41, 6), kept_backend=rec["kept_backend"], backend=rec["backend"], cpu=rec["cpu"], utime=rec["utime"], stime=rec["stime"], minflt=rec["minflt"],
                          wall=rec["wall"], cells=len(rec["cells"]), push_calls=sum(x[1] for x in push))), flush=True)
    json.dump(out, open(HERE / f"depth_shadow_{tag}.json", "w"))
print("DONE", tag, flush=True)
