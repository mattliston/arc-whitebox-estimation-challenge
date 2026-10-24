"""Bounded real-flopscope fixture for actual T48 native helper functions.

Physicist owns execution: one guarded single-thread process, 1.5GB reservation.
Only 1024-vectors and one 1024x128 projection are used; no model, network, truth
or captured bank is loaded. NumPy is offline fixture/reference code only.
Usage: python -B native_vector_fixture_v2.py --out NEW_REPORT.json
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
FRAGMENT_SHA = '06422496e5177c60a0a1a442c0fa448d97746fb7d7de4d5dda5190a807ab1eb9'
FUNCTIONS = {'_rph_begin', '_rph_const', '_rph_standardize', '_rph_diagonal', '_hc_bias'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.out)
    assert path.parent.is_dir() and not path.exists()
    # These imports occur only in the scheduled bounded fixture process.
    import numpy as np
    import flopscope as flops
    fnp = flops.numpy
    fragment = (HERE / 'fragment_repair_heads_v2.py').read_bytes()
    assert hashlib.sha256(fragment).hexdigest() == FRAGMENT_SHA
    nodes = [n for n in ast.parse(fragment).body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    assert {n.name for n in nodes} == FUNCTIONS
    code = compile(ast.Module(body=nodes, type_ignores=[]), '<exact native repair-head helpers>', 'exec')

    class Hold(list):
        def __call__(self, value):
            self.append(value)
            return value

    n = 1024
    variance = np.ones(n, dtype=np.float64)
    sigma = np.sqrt(variance)
    scale = variance * sigma
    rows = np.arange(64, dtype=np.int64)
    raw = np.zeros(n, dtype=np.float64)
    raw[rows] = np.arange(-32, 32, dtype=np.float64)
    dense_rows = np.arange(n, dtype=np.int64)
    cases = [
        ('missing', None, None),
        ('empty_roster', np.zeros(n), np.asarray([], dtype=np.int64)),
        ('empty_roster_nonzero_component', raw * scale, np.asarray([], dtype=np.int64)),
        ('zero_component', np.zeros(n), rows),
        ('constant_full', scale.copy(), dense_rows),
        ('sparse_with_exact_zero_inside', raw * scale, rows),
        ('invalid_std', np.full(n, np.nan), rows),
    ]

    def reference(repair, selected):
        result = np.zeros(n, dtype=np.float32)
        if repair is None:
            return result
        x = repair / scale
        centred = x - np.sum(x, dtype=np.float64) / n
        spread = np.sqrt(np.sum(centred * centred, dtype=np.float64) / (n - 1))
        if math.isfinite(float(spread)) and spread > 0:
            result[selected] = np.clip(centred[selected] / spread, -8, 8).astype(np.float32)
        return result

    weights = np.zeros((2, 128), dtype=np.float32)
    weights[0, 0], weights[1, 0] = .25, -.125
    weights[0, 2], weights[1, 5] = .001, -.002
    reports = []
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as ctx:
        namespace = dict(fnp=fnp, _E2Hold=Hold, _RPH_U=weights.tolist(),
                         _RPH_STATE={}, _RPH_YOUNG=[None], _RPH_CONST={})
        exec(code, namespace)
        ns = namespace
        def census():
            return int(ctx.summary_dict()['flops_used']), len(ctx._op_log)
        def delta(before):
            after = census()
            return dict(flops=after[0]-before[0], requests=after[1]-before[1])
        setup_before = census()
        ns['_rph_const'](n)
        setup = delta(setup_before)
        native_scale = fnp.asarray(scale, dtype=fnp.float64)
        for name, repair, selected in cases:
            native_repair = None if repair is None else fnp.asarray(repair, dtype=fnp.float64)
            native_rows = None if selected is None else fnp.asarray(selected, dtype=fnp.int64)
            before = census()
            result = ns['_rph_standardize'](native_repair, native_rows, native_scale, Hold())
            account = delta(before)
            got = np.asarray(result).copy()
            expected = reference(repair, selected)
            assert got.shape == (n,) and got.dtype == np.float32
            assert np.isfinite(got).all()
            assert np.allclose(got, expected, rtol=2e-6, atol=2e-7), (name, float(np.max(np.abs(got-expected))))
            outside = np.ones(n, dtype=bool)
            if selected is not None:
                outside[selected] = False
            assert np.array_equal(got[outside], np.zeros(int(outside.sum()), dtype=np.float32))
            if name == 'sparse_with_exact_zero_inside':
                assert got[32] != 0 and repair[32] == 0, 'actual roster must not become raw!=0 mask'
            reports.append(dict(case=name, maxabs=float(np.max(np.abs(got-expected))), **account))

        ownership_rejections = 0
        for invalid_repair, invalid_rows in ((None, rows), (raw, None), (raw[:4], rows)):
            invalid_repair = None if invalid_repair is None else fnp.asarray(invalid_repair, dtype=fnp.float64)
            invalid_rows = None if invalid_rows is None else fnp.asarray(invalid_rows, dtype=fnp.int64)
            try:
                ns['_rph_standardize'](invalid_repair, invalid_rows, native_scale, Hold())
            except RuntimeError:
                ownership_rejections += 1
        assert ownership_rejections == 3

        # Actual capture path: overlapping old/young rosters and signed repairs.
        old_np = raw * scale
        young_rows_np = np.arange(32, 128, dtype=np.int64)
        young_np = np.zeros(n, dtype=np.float64)
        young_np[young_rows_np] = -np.arange(1, 97, dtype=np.float64) * scale[young_rows_np]
        young_np[40] = -old_np[40]  # Same exact cancellation case as the Torch vector fixture.
        old = fnp.asarray(old_np, dtype=fnp.float64)
        young = fnp.asarray(young_np, dtype=fnp.float64)
        old_rows = fnp.asarray(rows, dtype=fnp.int64)
        young_rows = fnp.asarray(young_rows_np, dtype=fnp.int64)
        ns['_OTT_DATA'] = {'repair': old, 'full_tail': old_rows}
        ns['_YNT_REPAIR'] = [young]
        def original_diagonal(diagonal, lam):
            try:
                return (diagonal + ns['_OTT_DATA']['repair']) + ns['_YNT_REPAIR'][0]
            finally:
                ns['_YNT_REPAIR'][0] = None
        ns['_ott_diagonal'] = original_diagonal
        ns['_rph_begin'](7)
        ns['_RPH_YOUNG'][0] = young_rows
        diagonal_np = np.asarray([(i % 5) / 8 for i in range(n)], dtype=np.float64)
        diagonal = fnp.asarray(diagonal_np, dtype=fnp.float64)
        before = census()
        returned = ns['_rph_diagonal'](7, diagonal, None,
                                      fnp.asarray(variance, dtype=fnp.float64), fnp.asarray(sigma, dtype=fnp.float64))
        capture_account = delta(before)
        assert np.array_equal(np.asarray(returned), (diagonal_np + old_np) + young_np)
        assert ns['_YNT_REPAIR'][0] is None and ns['_RPH_YOUNG'][0] is None
        z = np.stack([reference(old_np, rows), reference(young_np, young_rows_np)])
        actual_z = np.stack([np.asarray(v).copy() for v in ns['_RPH_STATE']['columns']])
        assert np.allclose(actual_z, z, rtol=2e-6, atol=2e-7)
        assert old_np[40] + young_np[40] == 0
        assert actual_z[0, 40] != 0 and actual_z[1, 40] != 0, 'overlap components must remain separate'
        incumbent_np = np.full((n, 128), .0625, dtype=np.float32)
        incumbent = fnp.asarray(incumbent_np, dtype=fnp.float32)
        ns['_RPH_PARENT_HC_BIAS'] = lambda layer, bias: incumbent
        before = census()
        prediction = ns['_hc_bias'](7, incumbent)
        projection_account = delta(before)
        projected = np.asarray(prediction).copy()
        expected = incumbent_np + np.einsum('kn,kc->nc', actual_z, weights, optimize=False)
        assert projected.shape == (n, 128) and projected.dtype == np.float32
        assert np.allclose(projected, expected, rtol=2e-6, atol=2e-7)
        assert ns['_RPH_STATE'] == {} and ns['_RPH_YOUNG'][0] is None
        for layer in (6, 15):
            assert ns['_hc_bias'](layer, incumbent) is incumbent
        failed_closed = False
        try:
            ns['_hc_bias'](7, incumbent)
        except RuntimeError:
            failed_closed = True
        assert failed_closed
        namespace.clear()
    report = dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
                  status='PASS', scope='actual native helper vector/projection fixture only; no network/fit/gradient qualification',
                  source_sha256=FRAGMENT_SHA, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  setup=setup, normalization_cases=reports, overlap_capture=capture_account,
                  projection=projection_account, projection_shape=[1024,128],
                  signature='kn,kc->nc', operands=[[2,1024],[2,128]],
                  exact_original_diagonal=True, outside_roster_exact_zero=True,
                  actual_zero_inside_roster_retained=True, components_remain_separate=True,
                  state_consumed_once=True, missing_capture_fails_closed=True,
                  ownership_rejections=ownership_rejections, exact_cancellation_index=40,
                  normalization_floor_scope='vc and sigma supplied by native parent; source clamp is 1e-10')
    with path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'out':str(path),'status':'PASS','projection':projection_account}))


if __name__ == '__main__':
    main()
