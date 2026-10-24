#!/usr/bin/env python3
"""TAG fable-engineer (E-64): forward-time attribution of the S18-physics torch card by environment function (the native functions the
card runs through its numpy shim) and by shared kernel (push / read), one net (37), no grad and with grad (no backward).
usage (python with torch): refit_fn_timing.py <threads>"""
import sys, time, collections, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(0, str(P2 / 'fable_physicist_v1/refit6b')); sys.path.insert(0, str(HERE))
import numpy as np, torch
import s6bcard, s6bcard_j05
import lrstar_torch_hook as lh
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
p, a, c, e = s6bcard.payload(torch, params=P2 / 'fable_physicist_v1/refit6b/s6b_B2ep1_j05_1ep/params_epoch0.pt')
w = s6bcard.weights(37)
acc = collections.Counter(); calls = collections.Counter(); depth = [0]
def wrap(name, fn):
    def f(*args, **kw):
        t0 = time.perf_counter(); depth[0] += 1
        try: return fn(*args, **kw)
        finally:
            depth[0] -= 1; dt = time.perf_counter() - t0
            acc[name] += dt; calls[name] += 1
    return f
for grad in (False, True):
    acc.clear(); calls.clear()
    card = lh.install(s6bcard_j05.card(torch, w, p, a, c, e), torch)
    for k in list(card.env):
        if k in s6bcard.model.FUNCTIONS and callable(card.env[k]): card.env[k] = wrap(k, card.env[k])
    for k in list(getattr(card, 'shared', {})): card.shared[k] = wrap('shared.' + k, card.shared[k])
    t0 = time.perf_counter()
    if grad:
        for x in (p['head'], a, c): x.requires_grad_(True)
        pred = card.run(False)
    else:
        with torch.no_grad(): pred = card.run(False)
    tot = time.perf_counter() - t0
    print(json.dumps(dict(grad=grad, total_s=round(tot, 2), by_function={k: [round(v, 2), calls[k]] for k, v in acc.most_common()})), flush=True)
