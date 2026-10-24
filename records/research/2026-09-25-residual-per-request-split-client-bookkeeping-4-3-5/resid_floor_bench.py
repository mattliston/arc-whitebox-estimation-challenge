#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-RF1 v2 = v1 + pre-bound op, operators, .T and the bare fnp.add lookup; residual floor): flopscope's OWN client residual per request, measured at the client/server boundary
(numpy-free client + OpenBLAS server) with tight single-op loops and GC disabled. Residual = wall - flopscope's timed dispatch
(_dispatch.total_dispatch_ns), exactly the platform's definition. Per op type: us per iteration of wall, dispatch and residual,
requests (Connection.send_recv) and handle finalizers (_handles.enqueue_free) per iteration, and residual per request after the loop's
own Python (a pure-Python baseline loop) is subtracted. usage (client python, FLOPSCOPE_SERVER_URL set): resid_floor_bench.py <n>"""
import sys, time, gc, json
import flopscope as flops, flopscope.numpy as fnp
from flopscope import _dispatch, _connection, _handles
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
cnt = {"req": 0, "fin": 0}
_send = _connection.Connection.send_recv
def send(self, raw):
    cnt["req"] += 1; return _send(self, raw)
_connection.Connection.send_recv = send
_enq = _handles.enqueue_free
def enq(h):
    cnt["fin"] += 1; return _enq(h)
_handles.enqueue_free = enq
res = {}
with flops.BudgetContext(flop_budget=2**40, quiet=True, namespace="rfloor") as ctx:
    x = fnp.ones((4, 4), dtype=fnp.float32); y = fnp.ones((4, 4), dtype=fnp.float32); z = fnp.empty((4, 4), dtype=fnp.float32)
    keep = []
    ADD = fnp.add
    OPS = {
        "python_only": lambda: None,
        "add_new_result": lambda: fnp.add(x, y),
        "add_out_inplace": lambda: fnp.add(x, y, out=z),
        "getitem_view": lambda: x[1:3],
        "reshape_view": lambda: fnp.reshape(x, (16,)),
        "unstack4": lambda: fnp.unstack(x),
        "empty_alloc": lambda: fnp.empty((4, 4), dtype=fnp.float32),
        "add_kept_alive": lambda: keep.append(fnp.add(x, y)),
        "add_prebound": lambda: ADD(x, y),
        "operator_mul": lambda: x * y,
        "operator_matmul": lambda: x @ y,
        "attr_T": lambda: x.T,
        "lookup_only_fnp_add": lambda: fnp.add,
    }
    gc.disable()
    for name, fn in OPS.items():
        for _ in range(100): fn()
        keep.clear()
        r0, f0 = cnt["req"], cnt["fin"]; d0 = _dispatch.total_dispatch_ns(); t0 = time.perf_counter_ns()
        for _ in range(N): fn()
        t1 = time.perf_counter_ns(); d1 = _dispatch.total_dispatch_ns(); r1, f1 = cnt["req"], cnt["fin"]
        rec = dict(wall_us=(t1 - t0) / N / 1e3, dispatch_us=(d1 - d0) / N / 1e3, requests=(r1 - r0) / N, finalizers=(f1 - f0) / N)
        rec["resid_us"] = rec["wall_us"] - rec["dispatch_us"]; res[name] = rec
        if name == "add_kept_alive":   # the bulk release of N kept proxies, timed as residual per released handle
            d0 = _dispatch.total_dispatch_ns(); t0 = time.perf_counter_ns(); keep.clear(); t1 = time.perf_counter_ns(); d1 = _dispatch.total_dispatch_ns()
            res["bulk_release_per_handle"] = dict(resid_us=((t1 - t0) - (d1 - d0)) / N / 1e3, dispatch_us=(d1 - d0) / N / 1e3)
    gc.enable()
    s = ctx.summary_dict()
base = res["python_only"]["resid_us"]
print(f"N={N}; pure-Python loop residual {base:.3f} us/iter; context residual {float(s.get('residual_wall_time_s') or -1):.3f} s")
for k, v in res.items():
    if "requests" in v and v["requests"]:
        v["resid_per_request_us"] = (v["resid_us"] - base) / v["requests"]
    print(f"  {k:22s} " + " ".join(f"{a}={b:.3f}" for a, b in v.items()))
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
