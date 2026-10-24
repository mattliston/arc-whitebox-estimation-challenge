"""Root-only tiny paid FNP algebra fixture; no estimator import."""
from pathlib import Path
import ast, hashlib, json, os, sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent

def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

def main():
    assert len(sys.argv) == 3
    out = Path(sys.argv[2]).resolve(); assert not out.exists()
    build_pin = pin(D / 'build.json'); assert build_pin['sha256'] == sys.argv[1]
    build = json.loads((D / 'build.json').read_text())
    for key in ('parent', 'candidate', 'builder', 'fragment', 'primitive', 'protocol'):
        assert pin(build[key]['path']) == build[key]
    source = Path(build['candidate']['path']).read_text()
    fragment = Path(build['fragment']['path']).read_text()
    assert source.endswith('\n' + fragment)
    inverse = source[:-len('\n' + fragment)]
    for edit in reversed(build['edits']):
        assert inverse.count(edit['after']) == 1
        inverse = inverse.replace(edit['after'], edit['before'], 1)
    assert inverse == Path(build['parent']['path']).read_text()
    assert all(os.environ.get(key) == '1' for key in
               ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
    env = dict(fnp=fnp)
    exec(compile(fragment, build['fragment']['path'], 'exec'), env)
    n = 4
    C = np.array([[2., .25, -.125, .0625], [.25, 1.5, .375, -.25],
                  [-.125, .375, 1., .125], [.0625, -.25, .125, .75]])
    b = np.array([.0625, -.125, .1875, -.25])
    R32 = ((np.arange(n*n).reshape(n, n) % 7) - 3).astype(np.float32) / 16
    rd = np.array([.0312500001, -.0625000002, .0937500003, -.1250000004])
    cases = []

    def plane(covariance, vector):
        v = np.diag(covariance)
        return v[:, None] * vector[None, :] + 2 * covariance * vector[:, None]

    def run(name, covariance, residual, residual_diag):
        original = (covariance, b, residual, residual_diag)
        snapshots = [x.tobytes() for x in original]
        R = residual.astype(np.float64); np.fill_diagonal(R, residual_diag)
        # Explicit tiny linear design independently checks both cross terms,
        # transpose orientation and the separate high-precision diagonal.
        design = np.column_stack([plane(covariance, np.eye(n)[j]).reshape(-1) for j in range(n)])
        rhs = design.T @ R.reshape(-1)
        normal = design.T @ design
        p = rhs / np.maximum(np.diag(normal), 1e-30)
        alpha = (rhs @ p) / max(float(p @ normal @ p), 1e-30)
        expected_delta = .125 * alpha * p
        with flops.BudgetContext(flop_budget=100000, quiet=True) as budget:
            values = tuple(fnp.asarray(x) for x in original)
            paid_snapshots = [np.asarray(x).tobytes() for x in values]
            result = env['_bcls14'](*values)
            bb, rr, dd = [np.asarray(x).copy() for x in result]
            assert paid_snapshots == [np.asarray(x).tobytes() for x in values]
            bill = int(budget.summary_dict()['flops_used']); assert bill > 0
        assert snapshots == [x.tobytes() for x in original]
        assert all(np.isfinite(x).all() for x in (bb, rr, dd))
        assert np.allclose(bb-b, expected_delta, rtol=2e-12, atol=2e-12)
        assert np.allclose(dd, residual_diag-3*np.diag(covariance)*expected_delta, rtol=2e-12, atol=2e-12)
        corrected = rr.astype(np.float64); np.fill_diagonal(corrected, dd)
        before = plane(covariance, b) + R
        after = plane(covariance, bb) + corrected
        assert np.allclose(before, after, rtol=2e-7, atol=2e-8)
        assert np.linalg.norm(R-plane(covariance, expected_delta)) <= np.linalg.norm(R)+1e-12
        star = 3*rr.T - 2*np.diag(np.diag(rr)); np.fill_diagonal(star, dd.astype(np.float32))
        represented = star.T.astype(np.float64)/3; np.fill_diagonal(represented, np.diag(star))
        assert np.allclose(plane(covariance, bb)+represented, before, rtol=4e-7, atol=5e-8)
        if name in ('zero_residual', 'zero_covariance'):
            assert np.array_equal(bb, b)
        cases.append(dict(case=name, passed=True, flops=bill,
                          max_plane_error=float(np.max(np.abs(after-before)))))

    run('signed_symmetric', C, R32, rd)
    skew = C.copy(); skew[0, 1] += 1e-7
    run('rounded_nonsymmetric', skew, R32, rd)
    run('zero_residual', C, np.zeros_like(R32), np.zeros(n))
    run('zero_covariance', np.zeros_like(C), R32, rd)
    result = dict(passed=True, candidate=build['candidate'], build=build_pin,
                  probe=pin(__file__), cases=cases, all_inputs_unchanged=True,
                  full_tensor_equivalence_claimed=False, model_executed=False)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
