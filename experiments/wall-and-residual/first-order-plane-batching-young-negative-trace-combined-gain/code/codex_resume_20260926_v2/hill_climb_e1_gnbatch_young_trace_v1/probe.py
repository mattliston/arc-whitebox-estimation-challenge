"""Root-only exact-helper fixture: NATIVE -B probe.py BUILD_SHA RESULT."""
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
    assert build['parent']['sha256'] == '06538586dd306df2b00986f43cc85afd39d515a12c8977f4da6fa9c539ed91b1'
    ast.parse(source)
    env = dict(fnp=fnp, _OTT_CONTEXT=[None], _UVP_CUR=[7], _W5_PORT_SCOPE=object())
    calls = []; inject = [False]
    def push(w, p, previous, newborn):
        calls.append((int(w.shape[1]), env['_W5_PORT_SCOPE']))
        if inject[0]: raise RuntimeError('fixture parent failure')
        q, selected, base, t, s = newborn
        wp = w[previous, :].T
        common = wp @ base[previous, :]
        directions = w[selected, :].T
        return fnp.concatenate((wp @ p[:, :q], common + directions * t, common + directions * s), axis=1)
    def diagonal(d, lam): return d + fnp.asarray((np.arange(1024) % 3) * .125)
    def predict(fail=False):
        if fail: raise RuntimeError('fixture predict failure')
        return 'ok'
    env.update(_push_bank_shared_newborn=push, _ott_diagonal=diagonal, _predict_binary=predict)
    exec(compile(fragment, build['fragment']['path'], 'exec'), env)
    call, trace, diag = env['_push_bank_shared_newborn'], env['_ynt_trace'], env['_ott_diagonal']
    results = []
    for name, m, full_rows, q in [('full', 1024, None, 4), ('outer', 192, (np.arange(192) * 7) % 1024, 4),
                                 ('newborn_only', 192, (np.arange(192) * 7) % 1024, 0)]:
        n = 1024; previous = np.array([11, 2, 7, 29, 5], dtype=np.int64)
        selected = np.array([7, 11, 5], dtype=np.int64); b = len(selected)
        w = ((np.arange(n * m).reshape(n, m) % 17) - 8).astype(np.float32) / 16
        p = ((np.arange(len(previous) * q).reshape(len(previous), q) % 7) - 3).astype(np.float32) / 8
        B = np.zeros((n, b), np.float32)
        B[previous] = ((np.arange(len(previous) * b).reshape(len(previous), b) % 9) - 4) / 16
        B[selected, np.arange(b)] = 0
        t = np.array([.5, -.25, .125], np.float32); s = np.array([-.5, .125, -.25], np.float32)
        # Deliberately unequal signed pair magnitudes exercise actual two-tail ownership.
        coeff = np.concatenate((np.arange(q, dtype=np.float32) / 16 - .125,
                                np.array([.25, .5, .75, -.375, -.25, -.5], np.float32)))
        lam = np.concatenate((np.array([8., 9., 10.], np.float32), coeff))
        a = np.zeros(m, np.float64); a[-80:] = -3.0; a[-16:] = -4.0
        live = np.zeros(m, bool); live[:32] = True; live[-80:-64] = True
        negative = (a < -2.5) & ~live
        kept = min(m, ((np.count_nonzero(~negative) + 63) // 64) * 64)
        order = np.argsort(-np.where(negative, a + 2.5, 0), kind='stable')
        rows, tail = order[:kept], order[kept:]
        assert m - kept == 64 and not live[tail].any()
        columns = np.zeros((n, q + 2 * b), np.float64)
        columns[previous, :q] = p
        columns[:, q:q + b] = B; columns[:, q + b:] = B
        columns[selected, q + np.arange(b)] += t
        columns[selected, q + b + np.arange(b)] += s
        tau = columns @ (coeff.astype(np.float64) * np.sum(columns * columns, axis=0))
        expected = 3 * np.sum(w[:, tail].astype(np.float64) ** 2, axis=0) / (n + 2) * (w[:, tail].astype(np.float64).T @ tau)
        scope = object(); env['_W5_PORT_SCOPE'] = scope
        with flops.BudgetContext(flop_budget=100000000, quiet=True) as budget:
            ww, pp, prev, sel, bb, tt, ss, ll, aa, alive = map(fnp.asarray, (w, p, previous, selected, B, t, s, lam, a, live))
            physical = None if full_rows is None else fnp.asarray(full_rows)
            ctx = dict(a=aa, live=alive, rows=physical, n=n, lam=ll)
            env['_OTT_CONTEXT'][0] = ctx; newborn = (q, sel, bb, tt, ss)
            inputs = (ww, pp, prev, sel, bb, tt, ss, ll, aa, alive)
            before = [np.asarray(x).tobytes() for x in inputs]
            out = np.asarray(call(ww, pp, prev, newborn)).copy()
            assert calls[-1] == (kept, None) and env['_W5_PORT_SCOPE'] is scope
            expected_out = np.zeros_like(out)
            expected_out[rows] = np.asarray(push(ww[:, fnp.asarray(rows)], pp, prev, newborn))
            assert out.tobytes() == expected_out.tobytes()
            repair = np.asarray(env['_YNT_REPAIR'][0]).copy()
            indices = tail if full_rows is None else full_rows[tail]
            expected_repair = np.zeros(n); expected_repair[indices] = expected
            assert np.allclose(repair, expected_repair, rtol=2e-13, atol=2e-13)
            assert np.count_nonzero(repair[np.setdiff1d(np.arange(n), indices)]) == 0
            original = np.arange(n, dtype=np.float64) / 1024
            combined = np.asarray(diag(fnp.asarray(original), ll))
            assert np.allclose(combined, original + (np.arange(n) % 3) * .125 + expected_repair, rtol=2e-13, atol=2e-13)
            assert env['_YNT_REPAIR'][0] is None
            again = np.asarray(diag(fnp.asarray(original), ll))
            assert np.array_equal(again, original + (np.arange(n) % 3) * .125)
            assert before == [np.asarray(x).tobytes() for x in inputs]
            assert np.count_nonzero(B[np.setdiff1d(np.arange(n), previous)]) == 0
            for kind in ('inactive', 'no_context', 'no_block', 'all_alive'):
                env['_UVP_CUR'][0] = 6 if kind == 'inactive' else 7
                env['_OTT_CONTEXT'][0] = None if kind == 'no_context' else dict(ctx)
                if kind == 'no_block': env['_OTT_CONTEXT'][0]['a'] = fnp.asarray(np.r_[np.zeros(m - 32), np.full(32, -4.)])
                if kind == 'all_alive': env['_OTT_CONTEXT'][0]['live'] = fnp.asarray(np.ones(m, bool))
                fallback = np.asarray(call(ww, pp, prev, newborn))
                assert calls[-1] == (m, scope) and env['_YNT_REPAIR'][0] is None
                assert fallback.tobytes() == np.asarray(push(ww, pp, prev, newborn)).tobytes()
            env['_UVP_CUR'][0] = 7; env['_OTT_CONTEXT'][0] = ctx; inject[0] = True
            try: call(ww, pp, prev, newborn)
            except RuntimeError as e: assert str(e) == 'fixture parent failure'
            else: raise AssertionError('missing parent exception')
            finally: inject[0] = False
            assert env['_W5_PORT_SCOPE'] is scope and env['_YNT_REPAIR'][0] is None
            bill = int(budget.summary_dict()['flops_used']); assert bill > 0
        results.append(dict(case=name, flops=bill, passed=True, omitted_rows=64,
                            max_trace_error=float(np.max(np.abs(repair - expected_repair)))))
    for fail in (False, True):
        env['_YNT_REPAIR'][0] = 'stale'
        try: assert env['_predict_binary'](fail) == 'ok'
        except RuntimeError: assert fail
        assert env['_YNT_REPAIR'][0] is None and env['_YNT_STATS']['retired'] is True
    receipt = dict(passed=True, complete=True, candidate=build['candidate'], build=bp, probe=pin(__file__),
        cases=results, full_ambient_harmonic_scale=True, stored_signed_trace_verified=True,
        alive_outer_mapping_and_fallbacks=True, parent_scope_restored=True, correction_consumed_once=True,
        all_inputs_unchanged=True, prediction_finally_retired=True, no_predictor_parity_claim=True)
    with output.open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
