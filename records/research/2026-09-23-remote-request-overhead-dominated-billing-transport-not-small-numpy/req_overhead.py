#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-R): server CPU per flopscope REQUEST (numpy-free client + OpenBLAS server). Times N tiny requests of several
kinds (out= add on 8 floats; fresh add; getitem view; hidden free trip) and reports server CPU microseconds per request (ps cputime of
$E2_SERVER_PID) and client wall per request. usage: req_overhead.py [N]"""
import os, sys, time, subprocess
import flopscope as flops, flopscope.numpy as fnp
PID = os.environ['E2_SERVER_PID']; N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
def cpu():
    out = subprocess.run(['ps', '-o', 'cputime=', '-p', PID], capture_output=True, text=True).stdout.strip()
    sec = 0.0
    for x in out.replace('-', ':').split(':'):
        sec = sec * 60 + float(x)
    return sec
with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace="reqov"):
    a = fnp.ones(8); b = fnp.ones(8); c = fnp.empty(8); big = fnp.ones((64, 64)); hold = []
    for kind in ("add_out", "add_fresh_held", "getitem_held", "add_out_discard"):
        c0 = cpu(); t0 = time.perf_counter()
        for i in range(N):
            if kind == "add_out": hold.append(fnp.add(a, b, out=c))
            elif kind == "add_fresh_held": hold.append(fnp.add(a, b))
            elif kind == "getitem_held": hold.append(big[i % 64])
            else: fnp.add(a, b, out=c)          # discarded alias -> one hidden free trip per call (2 server requests)
        dt = time.perf_counter() - t0; time.sleep(1.2); dc = cpu() - c0 - 0.0
        hold.clear(); fnp.add(a, b, out=c)   # flush the held handles in one free trip
        print(f"{kind:16s} N={N}: server CPU {1e6 * dc / N:7.1f} us/call, client wall {1e6 * dt / N:7.1f} us/call", flush=True)
