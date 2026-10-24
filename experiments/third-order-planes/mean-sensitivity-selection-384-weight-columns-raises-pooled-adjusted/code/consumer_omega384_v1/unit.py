"""Actual selector/gather fixtures and independent mean-derivative check."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import types

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gaussian_mean(mu, v):
    alpha = mu / math.sqrt(v)
    return mu * (1 + math.erf(alpha / math.sqrt(2))) / 2 + math.sqrt(v / (2 * math.pi)) * math.exp(-alpha * alpha / 2)


def main():
    build = json.loads((HERE / 'build.json').read_text())
    assert sha(HERE / 'candidate.py') == build['candidate_sha256']
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    spec = importlib.util.spec_from_file_location('consumer_omega_unit', HERE / 'candidate.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    rng = np.random.Generator(np.random.PCG64(2026092014))
    variance = np.linspace(.25, 2., 1024)
    alpha = np.linspace(-8., 8., 1024)
    fixtures = [
        ('zero', np.zeros(1024), np.ones(1024)),
        ('equal', np.full(1024, .75), np.ones(1024)),
        ('mixed', alpha * np.sqrt(variance), variance),
        ('symmetric', np.tile(np.array([-.75, .75, -1., 1.]), 256), np.ones(1024)),
        ('clamped_tails', np.tile(np.array([-1e6, 0., 1e6, 1e-5]), 256),
         np.tile(np.array([1., 0., 1., -1.]), 256)),
        ('random', rng.normal(size=1024), np.exp(rng.normal(size=1024)))
    ]
    weights = rng.normal(size=(1024, 1024)).astype(np.float32)
    previous = rng.permutation(1024)[:896]
    rows = []
    rejects = 0
    with flops.BudgetContext(flop_budget=2**34, quiet=True) as ctx:
        spec.loader.exec_module(mod)
        w32, prev = fnp.asarray(weights), fnp.asarray(previous)
        for name, values, variances in fixtures:
            mu, vv = fnp.asarray(values), fnp.asarray(variances)
            safe = np.maximum(variances, 1e-10)
            standardized = values / np.sqrt(safe)
            score = np.abs(standardized) * np.exp(-standardized**2 / 2) / np.sqrt(2 * np.pi) / safe
            expected = np.argsort(-score, kind='stable')[:384]
            indices = mod._cs384_indices(mu, vv)
            repeat = mod._cs384_indices(mu, vv)
            reflected = mod._cs384_indices(-mu, vv)
            actual = np.asarray(indices)
            assert np.array_equal(actual, expected)
            assert actual.shape == (384,) and len(np.unique(actual)) == 384
            assert actual.min() >= 0 and actual.max() < 1024
            assert np.array_equal(np.asarray(repeat), actual)
            assert np.array_equal(np.asarray(reflected), actual)
            assert np.array_equal(np.asarray(mu), values) and np.array_equal(np.asarray(vv), variances)
            omega = mod._cs384_omega(w32, prev, indices)
            reference = weights[previous[:, None], expected[None, :]]
            assert np.asarray(omega).dtype == np.float32
            assert np.asarray(omega).tobytes() == reference.tobytes()
            if name in ('zero', 'equal'):
                assert np.array_equal(actual, np.arange(384))
                assert np.asarray(omega).tobytes() == np.asarray(w32[prev, :384]).tobytes()
            rows.append(dict(fixture=name, selected=int(len(actual)),
                             indices_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                             gather_sha256=hashlib.sha256(reference.tobytes()).hexdigest()))
        for wrong_mu, wrong_v in [(fnp.zeros(1023), fnp.ones(1024)),
                                  (fnp.zeros((1, 1024)), fnp.ones(1024)),
                                  (fnp.zeros(1024), fnp.ones(1023))]:
            try:
                mod._cs384_indices(wrong_mu, wrong_v)
            except RuntimeError:
                rejects += 1
            else:
                raise AssertionError('shape mismatch accepted')
        try:
            mod._cs384_omega(w32, prev, fnp.arange(383))
        except RuntimeError:
            rejects += 1
        else:
            raise AssertionError('column count accepted')
        assert np.array_equal(np.asarray(w32), weights)
        assert np.array_equal(np.asarray(prev), previous)
    derivative_rows = []
    for v in (.6, 1., 2.3):
        for standardized in (-3., -1., -.2, .2, 1., 3.):
            mu = standardized * math.sqrt(v)
            h = 1e-3 * math.sqrt(v)
            numeric = (gaussian_mean(mu+2*h, v)-2*gaussian_mean(mu+h, v)
                       +2*gaussian_mean(mu-h, v)-gaussian_mean(mu-2*h, v))/(2*h**3)
            analytic = -standardized * math.exp(-standardized**2/2)/(math.sqrt(2*math.pi)*v)
            error = abs(numeric-analytic)/max(1., abs(analytic))
            assert error <= 2e-5
            derivative_rows.append(dict(variance=v, alpha=standardized, scaled_error=error))
    assert rejects == 4
    assert mod._SCRATCH is None and mod._W5_PORT_SCOPE is None
    result = dict(passed=True, selector_rows=rows, derivative_rows=derivative_rows,
                  max_derivative_scaled_error=max(r['scaled_error'] for r in derivative_rows),
                  shape_rejections=rejects, exact_repeats_and_sign_symmetry=True,
                  input_preservation=True, gather_byte_exact=True,
                  zero_mean_parent_gather_byte_exact=True,
                  billed_flops=ctx.summary_dict()['flops_used'],
                  candidate_sha256=build['candidate_sha256'], source_sha256=sha(__file__),
                  truth_accessed=False, full_estimator_executed=False,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out = HERE / 'unit_v1'
    out.mkdir(exist_ok=False)
    with (out / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
