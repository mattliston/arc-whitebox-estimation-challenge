"""Root-only helper fixture: NATIVE -B probe.py BUILD_SHA RESULT."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def main():
    assert len(sys.argv) == 3
    output = Path(sys.argv[2]).resolve(); assert not output.exists()
    assert Path(sys.executable).absolute() == D.parent.parent / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
    assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
    bp = pin(D / 'build.json'); assert bp['sha256'] == sys.argv[1]
    build = json.loads((D / 'build.json').read_text())
    for key in ('parent', 'parent_build', 'candidate', 'builder', 'fragment', 'primitive', 'protocol', 'policy', 'native_driver'):
        assert pin(build[key]['path']) == build[key]
    parent = Path(build['parent']['path']).read_bytes(); source = Path(build['candidate']['path']).read_bytes()
    fragment = Path(build['fragment']['path']).read_bytes()
    assert source == parent + b'\n' + fragment and source[:-build['suffix_bytes']] == parent
    assert build['parent']['sha256'] == '88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
    ast.parse(source)
    env = dict(fnp=fnp, _OTT_CONTEXT=[None], _UVP_CUR=[7], _W5_PORT_SCOPE=object(), _PHYS_HANDLES={})
    calls = []; inject = [None]
    class Hold:
        def __init__(self): self.items = []
        def __call__(self, x): self.items.append(x); return x
    def mm(a, b):
        calls.append(('mm', tuple(a.shape), tuple(b.shape), env['_W5_PORT_SCOPE']))
        if inject[0] == 'inherited' and env['_W5_PORT_SCOPE'] is None:
            raise RuntimeError('fixture inherited failure')
        return a @ b
    def port(scope, w, previous, role, f, a, b):
        calls.append(('port', scope, w, previous, role, tuple(a.shape)))
        if inject[0] == 'newborn': raise RuntimeError('fixture newborn failure')
        return f(a, b)
    # Extract the exact final parent function, so fixture newborn parity is
    # against parent expressions rather than an independently rewritten helper.
    node = [n for n in ast.parse(parent).body if isinstance(n, ast.FunctionDef)
            and n.name == '_push_bank_shared_newborn'][-1]
    env.update(_E2Hold=Hold, _mm=mm, _w5_port_call=port)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned parent helper>', 'exec'), env)
    parent_push = env['_push_bank_shared_newborn']
    def diagonal(d, lam): return d + fnp.asarray((np.arange(1024) % 3) * .125)
    def predict(fail=False):
        if fail: raise RuntimeError('fixture predict failure')
        return 'ok'
    env.update(_ott_diagonal=diagonal, _predict_binary=predict)
    exec(compile(fragment, build['fragment']['path'], 'exec'), env)
    call, diag = env['_push_bank_shared_newborn'], env['_ott_diagonal']; results = []
    for name, m, full_rows, sign in [('full', 1024, None, 1),
            ('outer', 192, (np.arange(192) * 7) % 1024, 1),
            ('signed', 192, (np.arange(192) * 7) % 1024, -1)]:
        n, q = 1024, 4; previous = np.array([11, 2, 7, 29, 5], np.int64)
        selected = np.array([7, 11, 5], np.int64); b = len(selected)
        w = ((np.arange(n * m).reshape(n, m) % 17) - 8).astype(np.float32) / 16
        p = ((np.arange(len(previous) * q).reshape(len(previous), q) % 7) - 3).astype(np.float32) / 8
        B = np.zeros((n, b), np.float32)
        B[previous] = ((np.arange(len(previous) * b).reshape(len(previous), b) % 9) - 4) / 16
        B[selected, np.arange(b)] = 0
        t = np.array([.5, -.25, .125], np.float32); s = np.array([-.5, .125, -.25], np.float32)
        coeff = sign * np.array([.125, -.25, .5, -.375], np.float32)
        newborn_coeff = np.array([.25, .5, .75, -.375, -.25, -.5], np.float32)
        lam = np.concatenate((np.array([8., 9., 10.], np.float32), coeff, newborn_coeff))
        a = np.zeros(m); a[-80:] = -3.; a[-16:] = -4.
        live = np.zeros(m, bool); live[:32] = True; live[-80:-64] = True
        negative = (a < -2.5) & ~live
        kept = min(m, ((np.count_nonzero(~negative) + 63) // 64) * 64)
        order = np.argsort(-np.where(negative, a + 2.5, 0), kind='stable')
        rows, tail = order[:kept], order[kept:]
        assert m - kept == 64 and not live[tail].any()
        embedded = np.zeros((n, q)); embedded[previous] = p
        tau = embedded @ (coeff.astype(np.float64) * np.sum(embedded * embedded, axis=0))
        expected = 3 * np.sum(w[:, tail].astype(np.float64) ** 2, axis=0) / (n + 2) * (w[:, tail].astype(np.float64).T @ tau)
        scope = object(); env['_W5_PORT_SCOPE'] = scope
        with flops.BudgetContext(flop_budget=100000000, quiet=True) as budget:
            ww, pp, prev, sel, bb, tt, ss, ll, aa, alive = map(fnp.asarray, (w, p, previous, selected, B, t, s, lam, a, live))
            physical = None if full_rows is None else fnp.asarray(full_rows)
            ctx = dict(a=aa, live=alive, rows=physical, n=n, lam=ll)
            env['_OTT_CONTEXT'][0] = ctx; newborn = (q, sel, bb, tt, ss)
            inputs = (ww, pp, prev, sel, bb, tt, ss, ll, aa, alive)
            before = [np.asarray(x).tobytes() for x in inputs]
            baseline = np.asarray(parent_push(ww, pp, prev, newborn)).copy()
            calls.clear(); out = np.asarray(call(ww, pp, prev, newborn)).copy()
            assert calls[0] == ('mm', (kept, len(previous)), (len(previous), q), None)
            pc = [v for v in calls if v[0] == 'port']; assert len(pc) == 1
            assert pc[0][1] is scope and pc[0][2] is ww and pc[0][3] is prev
            assert pc[0][4:] == ('newborn_common', (m, len(previous)))
            assert env['_W5_PORT_SCOPE'] is scope
            assert out[:, q:].tobytes() == baseline[:, q:].tobytes()
            expected_out = baseline.copy(); expected_out[tail, :q] = 0
            assert out.tobytes() == expected_out.tobytes()
            repair = np.asarray(env['_YNT_REPAIR'][0]).copy()
            indices = tail if full_rows is None else full_rows[tail]
            expected_repair = np.zeros(n); expected_repair[indices] = expected
            assert np.allclose(repair, expected_repair, rtol=2e-13, atol=2e-13)
            assert np.count_nonzero(repair[np.setdiff1d(np.arange(n), indices)]) == 0
            changed = lam.copy(); changed[-2 * b:] *= 8
            ctx['lam'] = fnp.asarray(changed)
            assert np.asarray(call(ww, pp, prev, newborn)).tobytes() == out.tobytes()
            assert np.asarray(env['_YNT_REPAIR'][0]).tobytes() == repair.tobytes()
            ctx['lam'] = ll
            original = np.arange(n, dtype=np.float64) / 1024
            combined = np.asarray(diag(fnp.asarray(original), ll))
            assert np.allclose(combined, original + (np.arange(n) % 3) * .125 + expected_repair, rtol=2e-13, atol=2e-13)
            assert env['_YNT_REPAIR'][0] is None
            assert np.array_equal(np.asarray(diag(fnp.asarray(original), ll)), original + (np.arange(n) % 3) * .125)
            for kind in ('inactive', 'no_context', 'no_block', 'all_alive', 'no_inherited'):
                env['_UVP_CUR'][0] = 6 if kind == 'inactive' else 7
                env['_OTT_CONTEXT'][0] = None if kind == 'no_context' else dict(ctx)
                if kind == 'no_block': env['_OTT_CONTEXT'][0]['a'] = fnp.asarray(np.r_[np.zeros(m - 32), np.full(32, -4.)])
                if kind == 'all_alive': env['_OTT_CONTEXT'][0]['live'] = fnp.asarray(np.ones(m, bool))
                parg, nb = (pp[:, :0], (0, sel, bb, tt, ss)) if kind == 'no_inherited' else (pp, newborn)
                fallback = np.asarray(call(ww, parg, prev, nb)).copy()
                assert env['_YNT_REPAIR'][0] is None and env['_W5_PORT_SCOPE'] is scope
                assert fallback.tobytes() == np.asarray(parent_push(ww, parg, prev, nb)).tobytes()
            env['_UVP_CUR'][0] = 7; env['_OTT_CONTEXT'][0] = ctx
            for stage in ('inherited', 'newborn'):
                inject[0] = stage
                try: call(ww, pp, prev, newborn)
                except RuntimeError as e: assert str(e) == 'fixture ' + stage + ' failure'
                else: raise AssertionError('missing parent exception')
                finally: inject[0] = None
                assert env['_W5_PORT_SCOPE'] is scope and env['_YNT_REPAIR'][0] is None
            assert before == [np.asarray(x).tobytes() for x in inputs]
            bill = int(budget.summary_dict()['flops_used']); assert bill > 0
        results.append(dict(case=name, flops=bill, passed=True, omitted_rows=64,
            newborn_bytes_exact=True, max_trace_error=float(np.max(np.abs(repair - expected_repair)))))
    for fail in (False, True):
        env['_YNT_REPAIR'][0] = 'stale'
        try: assert env['_predict_binary'](fail) == 'ok'
        except RuntimeError: assert fail
        assert env['_YNT_REPAIR'][0] is None and env['_YNT_STATS']['retired'] is True
    receipt = dict(passed=True, complete=True, candidate=build['candidate'], build=bp, probe=pin(__file__),
        cases=results, full_ambient_harmonic_scale=True, inherited_signed_trace_verified=True,
        newborn_block_bytes_exact=True, newborn_coefficients_excluded_from_trace=True,
        alive_outer_mapping_and_fallbacks=True, newborn_scope_and_operands_exact=True,
        correction_consumed_once=True, all_inputs_unchanged=True, prediction_finally_retired=True,
        no_predictor_parity_claim=True)
    with output.open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
