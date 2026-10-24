"""[opus-callform] client-server micro-bench of call FORMS: per call, client instructions in total, inside flopscope's dispatch spans
(the boundary_client accounting: counter reads inside the outermost span) and the RESIDUAL (= total - dispatch: what the platform bills
as participant Python), plus server requests. Run with the release CLIENT_PY (numpy-free flopscope client) against server_light.
usage: mb_client_v1.py <out.json> [N]   env: FLOPSCOPE_SERVER_URL"""
import json, sys, time
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'release')); import rusage
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _connection, _dispatch
out = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
count = dict(requests=0)
_send = _connection.Connection.send_recv
def counted_send(self, raw):
    count['requests'] += 1
    return _send(self, raw)
_connection.Connection.send_recv = counted_send
_dispatch.__dict__.update(_release_raw=rusage.raw, _release_depth=[0], _release_acc=[0, 0], _release_span=_dispatch._counted_span)
exec('''
@contextmanager
def _counted_span():
    with _release_span():
        top = _release_depth[0] == 0
        if top:
            c0, i0 = _release_raw()
        _release_depth[0] += 1
        try:
            yield
        finally:
            _release_depth[0] -= 1
            if top:
                c1, i1 = _release_raw()
                _release_acc[0] += c1 - c0
                _release_acc[1] += i1 - i0
''', _dispatch.__dict__)
ACC = _dispatch._release_acc
res = {}
with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace='callform'):
    import random
    a = fnp.asarray([[random.random() for _ in range(64)] for _ in range(64)], dtype=fnp.float32)
    b = fnp.asarray([[random.random() for _ in range(64)] for _ in range(64)], dtype=fnp.float32)
    c = fnp.empty((64, 64), dtype=fnp.float32)
    z0 = fnp.asarray(2.0, dtype=fnp.float32)
    FORMS = {
        'baseline_noop': lambda h: h.append(a),
        'mul_operator': lambda h: h.append(a * b),
        'mul_function': lambda h: h.append(fnp.multiply(a, b)),
        'mul_out_kw': lambda h: h.append(fnp.multiply(a, b, out=c)),
        'mul_out_pos': lambda h: h.append(fnp.multiply(a, b, c)),
        'add_operator': lambda h: h.append(a + b),
        'add_function': lambda h: h.append(fnp.add(a, b)),
        'add_out_kw': lambda h: h.append(fnp.add(a, b, out=c)),
        'mul_pyscalar': lambda h: h.append(a * 2.0),
        'mul_0d_array': lambda h: h.append(a * z0),
        'mul_fn_pyscalar': lambda h: h.append(fnp.multiply(a, 2.0)),
        'einsum_ij': lambda h: h.append(fnp.einsum('ij,ij->ij', a, b, optimize=False)),
        'getitem_int': lambda h: h.append(a[3]),
        'getitem_slice': lambda h: h.append(a[:32]),
        'unstack_64': lambda h: h.append(fnp.unstack(a)),
        'split_4': lambda h: h.append(fnp.split(a, 4)),
        'reshape_fn': lambda h: h.append(fnp.reshape(a, (8, 8, 64))),
        'reshape_method': lambda h: h.append(a.reshape(8, 8, 64)),
        'transpose_T': lambda h: h.append(a.T),
        'transpose_fn': lambda h: h.append(fnp.transpose(a)),
        'mul_discard': lambda h: fnp.multiply(a, b),
        'sum_axis': lambda h: h.append(fnp.sum(a, axis=0)),
        'matmul_op': lambda h: h.append(a @ b),
        'matmul_fn': lambda h: h.append(fnp.matmul(a, b)),
    }
    for name, fn in FORMS.items():
        for rep in range(2):   # rep 0 warms up; rep 1 is measured
            hold = []
            r0, (cy0, in0), d0 = count['requests'], rusage.raw(), list(ACC)
            for i in range(N):
                fn(hold)
                if i % 100 == 99:
                    hold.clear()   # frees in batches, as _E2Hold does (the free path counts as residual too)
            cy1, in1 = rusage.raw(); r1 = count['requests']
            disp = ACC[1] - d0[1]; tot = in1 - in0
            if rep:
                res[name] = dict(n=N, requests_per_call=(r1 - r0) / N, total_instr_per_call=tot / N, dispatch_instr_per_call=disp / N,
                                 residual_instr_per_call=(tot - disp) / N)
            del hold
base = res['baseline_noop']['residual_instr_per_call']
for k, v in res.items():
    v['residual_minus_loop'] = v['residual_instr_per_call'] - base
out.write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(f"{k:18s} req/call {v['requests_per_call']:.2f}  total {v['total_instr_per_call']:9.0f}  dispatch {v['dispatch_instr_per_call']:9.0f}  residual {v['residual_instr_per_call']:8.0f}  (minus loop {v['residual_minus_loop']:8.0f})")
