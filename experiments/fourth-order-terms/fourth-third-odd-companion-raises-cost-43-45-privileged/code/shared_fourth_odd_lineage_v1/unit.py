"""Independent tensor aggregation, actual FNP transport, and scalar oracle checks."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import resource
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from feature_fnp import (advance_u, epsilon_from_aux, gate_u, mean_feature,
                         observed_push, odd_diagonal, pair_gamma)
from oracle import clipped_scalar_oracle

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'lazy_rotation_objective_v1/candidate.py'
PARENT_SHA = 'b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual)-expected))/max(1., np.max(np.abs(expected))))


def sym(u, p, coeff):
    one = np.einsum('ir,jr,kr,r->ijk', u, p, p, coeff)
    return one+one.transpose(1, 0, 2)+one.transpose(1, 2, 0)


def exhaustive_masks(p, z, y):
    """Independent small reference enumerating every feasible active set."""
    best = np.inf
    for mask_tuple in itertools.product((False, True), repeat=len(p)):
        mask = np.array(mask_tuple)
        low, high = -np.inf, np.inf
        possible = True
        for pi, zi, active in zip(p, z, mask):
            if zi == 0:
                possible &= (pi >= 0 if active else pi <= 0)
            else:
                boundary = -pi/zi
                if (zi > 0) == active:
                    low = max(low, boundary)
                else:
                    high = min(high, boundary)
        if not possible or low > high:
            continue
        a = sum(float(zi*zi) for zi, active in zip(z, mask) if active)
        b = sum(float(zi*(pi-yi)) for pi, zi, yi, active in zip(p, z, y, mask) if active)
        coefficient = np.clip(-b/a if a else 0., low, high)
        best = min(best, float(np.mean((np.maximum(p+coefficient*z, 0.)-y)**2)))
    return best


def main():
    out = HERE/'unit_v1'
    out.mkdir(exist_ok=False)
    seal = json.loads((HERE/'sources_sealed.json').read_text())
    for name, digest in seal['artifacts_sha256'].items():
        assert sha(HERE/name) == digest, name
    assert sha(PARENT) == PARENT_SHA
    observed, proof = observed_push(PARENT.read_text())
    assert proof['inverse_byte_exact']
    rng = np.random.Generator(np.random.PCG64(2026092016))
    n, pairs = 7, 4
    coeff = rng.normal(size=2*pairs)
    p = rng.normal(size=(n, 2*pairs))*.3
    u = np.zeros_like(p)
    tensor = np.zeros((n, n, n))
    dense_errors = []
    for step in range(4):
        matrix = rng.normal(size=(n, n))/np.sqrt(n)
        pp, up = matrix@p, matrix@u
        pre = np.einsum('ai,bj,ck,ijk->abc', matrix, matrix, matrix, tensor)
        g = rng.uniform(.1, .9, n)
        d = rng.uniform(.05, .6, n)
        alive = np.ones(n)
        alive[step] = 0
        h, derivative = g*alive, d*alive
        tensor = pre*h[:, None, None]*h[None, :, None]*h[None, None, :]
        tensor += .5*sym(derivative[:, None]*pp**2, h[:, None]*pp, coeff)
        p = h[:, None]*pp
        u = h[:, None]*up+derivative[:, None]*pp**2
        dense_errors.append(error(.5*sym(u, p, coeff), tensor))
        dense_errors.append(error(1.5*np.sum(u*p*p*coeff[None, :], axis=1), np.einsum('iii->i', tensor)))
    assert max(dense_errors) < 1e-12

    oracle_errors = []
    fixtures = [(np.array([1., 2., .2]), np.zeros(3), np.array([.5, 1.2, .4])),
                (np.zeros(4), np.array([1., -1., 2., -2.]), np.array([.7, 0., 1.4, 0.])),
                (np.array([1., 1., 2.]), np.array([-1., -1., -2.]), np.zeros(3)),
                (np.ones(4), np.array([1., 0., -1., 2.]), np.array([1.2, 1., .8, 1.4]))]
    for unused in range(12):
        fixtures.append((rng.uniform(0., 2., 6), rng.normal(size=6), rng.uniform(0., 2., 6)))
    for parent, feature, truth in fixtures:
        result, ideal, served = clipped_scalar_oracle(parent, feature, truth)
        reference = exhaustive_masks(parent, feature, truth)
        oracle_errors.append(abs(result['ideal_raw']-reference))
        assert oracle_errors[-1] < 1e-12
        grid = np.linspace(-10., 10., 20001)
        grid_error = np.min(np.mean((np.maximum(parent+grid[:, None]*feature, 0.)-truth)**2, axis=1))
        assert result['ideal_raw'] <= grid_error+1e-12
        assert result['rounding_allowance_lower_raw'] <= result['served_raw']+1e-20
        assert np.array_equal(served, ideal.astype(np.float32))

    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    spec = importlib.util.spec_from_file_location('odd_unit_parent', PARENT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    helper_errors, packed_checks = [], []
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        spec.loader.exec_module(mod)
        alpha = np.linspace(-2., 2., n)
        sigma = np.linspace(.3, 2., n)
        pdf = np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)
        aux = {k:fnp.asarray(v) for k, v in dict(alpha=alpha, sigma=sigma, pdf=pdf).items()}
        selected = np.array([0, 2, 4, 6])
        t = np.array([.7, .3, .5, .125], dtype=np.float32)
        s = np.array([-.1, .3, -.2, -.125], dtype=np.float32)
        eps = -alpha*pdf/sigma**2
        gap = t.astype(np.float64)-s.astype(np.float64)
        expected_gamma = np.where(gap != 0, eps[selected]/np.where(gap != 0, gap, 1.), 0.)
        gamma = pair_gamma(epsilon_from_aux(aux), fnp.asarray(selected), fnp.asarray(t), fnp.asarray(s))
        assert np.array_equal(np.asarray(gamma), expected_gamma)
        pp = rng.normal(size=(n, 8)).astype(np.float32)
        up = rng.normal(size=(n, 8)).astype(np.float32)
        g = rng.uniform(.1, .9, n)
        d = rng.uniform(.05, .6, n)
        alive = np.array([True, False, True, True, True, False, True])
        pp_f, up_f = fnp.asarray(pp), fnp.asarray(up.copy())
        actual = gate_u(up_f, pp_f, fnp.asarray(g), fnp.asarray(d), fnp.asarray(alive))
        expected = alive[:, None]*(g[:, None]*up.astype(np.float64)+d[:, None]*pp.astype(np.float64)**2)
        helper_errors.append(error(actual, expected))
        repeated = gate_u(fnp.asarray(up.copy()), pp_f, fnp.asarray(g), fnp.asarray(d), fnp.asarray(alive))
        assert np.asarray(actual).tobytes() == np.asarray(repeated).tobytes()
        assert np.asarray(pp_f).tobytes() == pp.tobytes()
        cohorts = [(0, gamma[:2]), (4, gamma[2:])]
        coefficients = np.concatenate((expected_gamma[:2], -expected_gamma[:2], expected_gamma[2:], -expected_gamma[2:]))
        dense = .5*sym(np.asarray(actual).astype(np.float64), pp.astype(np.float64), coefficients)
        diagonal = odd_diagonal(actual, pp_f, cohorts)
        scalar_error = error(diagonal, np.einsum('iii->i', dense))
        assert scalar_error < 1e-10
        expected_feature = -alpha*pdf*np.einsum('iii->i', dense)/(6*sigma**2)
        feature_error = error(mean_feature(aux, diagonal), expected_feature)
        assert feature_error < 1e-10
        assert np.count_nonzero(np.asarray(odd_diagonal(actual, pp_f, [(0, gamma*0)]))) == 0

        # Small actual helper, including ordinary, fresh projection, terminal
        # direct push, and a zero-column initial layer. No packed kernel here.
        w = rng.normal(size=(n, n)).astype(np.float32)
        previous = np.array([0, 2, 3, 4, 6])
        u0 = rng.normal(size=(n, 6)).astype(np.float32)
        q = np.linalg.qr(rng.normal(size=(len(previous), 3)))[0].astype(np.float32)
        pre_basis = (w[previous, :].T.astype(np.float64)@q.astype(np.float64)).astype(np.float32)
        dummy = fnp.zeros((n, 8), dtype=fnp.float32)
        for layer in (3, 5, 15):
            pushed = advance_u(lambda a, b:a@b, layer, fnp.asarray(w), fnp.asarray(previous),
                               fnp.asarray(u0), dummy, 4, fnp.asarray(q), fnp.asarray(pre_basis))
            if layer == 5:
                old = pre_basis.astype(np.float64)@(q.astype(np.float64).T@u0[previous, :4].astype(np.float64))
                young = w[previous, :].T.astype(np.float64)@u0[previous, 4:].astype(np.float64)
                reference = np.concatenate((old, young, np.zeros((n, 2))), axis=1)
            else:
                reference = np.concatenate((w[previous, :].T.astype(np.float64)@u0[previous, :].astype(np.float64), np.zeros((n, 2))), axis=1)
            helper_errors.append(error(pushed, reference))
        empty = advance_u(lambda a, b:a@b, 0, fnp.asarray(w), fnp.asarray(previous),
                          fnp.zeros((n, 0), fnp.float32), fnp.zeros((n, 0), fnp.float32), 0, None, None)
        assert empty.shape == (n, 0)
        assert max(helper_errors) < 1e-5

        arena = fnp.empty((mod._PACKED_ELEMENTS,), dtype=fnp.float32)
        mod._SCRATCH = [arena, mod._cp3_bind_arena_views(arena, mod._PACKED_PUSH),
                        mod._cp3_bind_arena_views(arena, mod._PACKED_READ), None, None, None, None]
        mod._bind_push6(arena)
        for m, k, width in ((384, 896, 2048), (1024, 384, 2048), (1024, 896, 2048)):
            a = (rng.normal(size=(m, k))/np.sqrt(k)).astype(np.float32)
            b = rng.normal(size=(k, width)).astype(np.float32)
            a_f, b_f = fnp.asarray(a), fnp.asarray(b)
            before = ctx.summary_dict()['flops_used']
            got = mod._shared_old384_mm(a_f, b_f)
            paid = ctx.summary_dict()['flops_used']-before
            copy = np.asarray(got).copy()
            reference = a.astype(np.float64)@b.astype(np.float64)
            relative = float(np.linalg.norm(copy.astype(np.float64)-reference)/np.linalg.norm(reference))
            repeated = mod._shared_old384_mm(a_f, b_f)
            assert relative <= 2e-4, (m, k, width, relative)
            assert copy.tobytes() == np.asarray(repeated).tobytes()
            assert np.asarray(a_f).tobytes() == a.tobytes() and np.asarray(b_f).tobytes() == b.tobytes()
            packed_checks.append(dict(shape=[m, k, width], relative_error=relative, flops=paid,
                                      repeat_byte_exact=True, inputs_unchanged=True))
            del a, b, a_f, b_f, got, copy, reference, repeated
    result = dict(passed=True, dense_multitransition_max_error=max(dense_errors),
                  actual_helper_max_error=max(helper_errors), scalar_error=scalar_error,
                  feature_error=feature_error, zero_gamma_exact=True, zero_gap_exact=True,
                  oracle_fixtures=len(fixtures), oracle_max_error=max(oracle_errors),
                  packed_products=packed_checks, observed_source=proof,
                  full_estimator_executed=False, truth_accessed=False,
                  flops=ctx.summary_dict()['flops_used'], source_sha256=sha(__file__),
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (out/'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
