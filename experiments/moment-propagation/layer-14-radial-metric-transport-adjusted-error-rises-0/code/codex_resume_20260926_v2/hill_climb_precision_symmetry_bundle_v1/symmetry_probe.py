"""Root-only fixed isolated symmetry fixtures; synthetic inputs, no model labels."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import types
import numpy as np
import flopscope as flops

D = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summary(ctx):
    value = ctx.summary_dict()
    return dict(flops=int(value['flops_used']), operations=sum(int(v.get('calls', v.get('count', 0))) for v in value['operations'].values()))


def main():
    seal = json.loads((D/'build.json').read_text())
    for rec in seal['files']:
        assert sha(rec['path']) == rec['sha256'], rec['path']
    assert seal['candidate']['sha256'] == sha(D/'candidate.py')
    assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'))
    assert np.__version__ == '2.4.6'
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    mod = types.ModuleType('symmetry_candidate')
    mod.__file__ = str(D/'candidate.py')
    sys.modules[mod.__name__] = mod
    exec(compile((D/'candidate.py').read_text(), mod.__file__, 'exec'), mod.__dict__)
    t0 = time.monotonic()
    mod.Estimator().setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
    setup_s = time.monotonic()-t0
    fnp = mod.fnp
    records = []
    for kind, seed in (('integer_gram', 202609285), ('gaussian_covariance', 202609286)):
        rng = np.random.default_rng(seed)
        if kind == 'integer_gram':
            w = (2*rng.integers(0, 2, size=(1024, 1024))-1).astype(np.float32)
            a, b = w.T, w.copy()
        else:
            w = (rng.standard_normal((1024, 1024))/32).astype(np.float32)
            h = (rng.standard_normal((1024, 1024))/32).astype(np.float32)
            cov = h @ h.T
            # Both methods receive the same finite precision stage-one product.
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                cw = mod._mm(fnp.asarray(cov), fnp.asarray(w))
            stage_one = summary(ctx)
            a, b = w.T, np.asarray(cw).copy()
            del h, cov, cw
        reference = a.astype(np.float64) @ b.astype(np.float64)
        inputs = [hashlib.sha256(v.tobytes()).hexdigest() for v in (a, b)]
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            aa, bb = fnp.asarray(a), fnp.asarray(b)
        conversions = summary(ctx)
        runs = []
        for label, method in (('parent_first', mod._PS_PARENT_SYM_UPPER), ('child_first', mod._ps_sym_upper), ('parent_repeat', mod._PS_PARENT_SYM_UPPER), ('child_repeat', mod._ps_sym_upper)):
            mod._TR_LAST[0] = None
            t0, c0 = time.monotonic(), time.process_time()
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                out = method(aa, bb)
            wall, cpu = time.monotonic()-t0, time.process_time()-c0
            value = np.asarray(out).copy()
            assert value.shape == (1024, 1024) and value.dtype == np.float32 and np.isfinite(value).all()
            relative = float(np.linalg.norm(value.astype(np.float64)-reference)/np.linalg.norm(reference))
            exact = bool(np.array_equal(value.astype(np.float64), reference))
            assert exact if kind == 'integer_gram' else relative <= 2e-4
            assert inputs == [hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in (aa, bb)]
            runs.append(dict(label=label, wall_s=wall, cpu_s=cpu, relative_frobenius_error=relative, exact=exact, output_sha256=hashlib.sha256(value.tobytes()).hexdigest(), **summary(ctx)))
            del out, value
        child_runs = [r for r in runs if r['label'].startswith('child')]
        assert child_runs[0]['output_sha256'] == child_runs[1]['output_sha256']
        rec = dict(kind=kind, seed=seed, input_sha256=inputs, input_strides=[list(v.strides) for v in (a, b)], conversions=conversions, runs=runs, binding_cold_case=(kind == 'integer_gram'))
        if kind != 'integer_gram': rec['stage_one'] = stage_one
        records.append(rec)
        print(json.dumps(dict(event='case_complete', **rec)), flush=True)
        del a, b, aa, bb, reference, w
    result = dict(passed=True, source_sha256=sha(D/'candidate.py'), protocol_sha256=sha(D/'PROTOCOL.md'), build_sha256=sha(D/'build.json'), probe_sha256=sha(__file__), setup_s=setup_s, cases=records, model_executed=False, network_targets_read=False, serving_qualified=False)
    with (D/'symmetry_result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('SYMMETRY_FIXTURES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
