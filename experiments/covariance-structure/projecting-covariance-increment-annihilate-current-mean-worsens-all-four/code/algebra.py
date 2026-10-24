"""Root-only tiny check of the actual native helper; no estimator imports."""
from pathlib import Path
import ast, hashlib, json, sys

D = Path(__file__).resolve().parent


def main():
    assert len(sys.argv) == 2
    path = D/'fragment.py'; source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest(); assert digest == sys.argv[1]
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gn_perp_host']
    assert len(functions) == 1
    import numpy as np
    env = {'fnp': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), env)
    project = env['_gn_perp_host']; rows = []
    for n in (3, 6, 9):
        x = np.arange(n*n, dtype=np.float32).reshape(n, n)
        increment = (x % 13 - 6) / np.float32(8)
        means = [np.ones(n), 1.0 + np.arange(n)*.03,
                 (1.0 + np.arange(n)*.03)*np.where(np.arange(n)%2, -1., 1.)]
        for mean in means:
            saved = increment.tobytes(), mean.tobytes()
            got = project(10, mean, increment)
            A = (increment.astype(np.float64) + increment.astype(np.float64).T)*.5
            np.fill_diagonal(A, 0.)
            edges = [(i,j) for i in range(n) for j in range(i+1,n)]
            constraint = np.zeros((n,len(edges)))
            target = np.array([A[i,j] for i,j in edges])
            for k,(i,j) in enumerate(edges):
                constraint[i,k] = mean[j]; constraint[j,k] = mean[i]
            dual = np.linalg.lstsq(constraint@constraint.T, constraint@target, rcond=None)[0]
            solution = target - constraint.T@dual
            expected = np.zeros((n,n))
            for value,(i,j) in zip(solution,edges): expected[i,j] = expected[j,i] = value
            np.testing.assert_allclose(got,expected,rtol=2e-11,atol=2e-12)
            assert np.array_equal(got,got.T) and np.count_nonzero(np.diag(got)) == 0
            assert float(np.max(np.abs(got@mean))) < 2e-11
            np.testing.assert_allclose(project(11,mean,got),got,rtol=2e-11,atol=2e-12)
            np.testing.assert_allclose(project(12,mean*7,increment),got,rtol=2e-11,atol=2e-12)
            assert (increment.tobytes(),mean.tobytes()) == saved
            rows.append(dict(n=n,max_dense_error=float(np.max(np.abs(got-expected))),
                             mean_residual=float(np.max(np.abs(got@mean)))))
        assert project(9,np.ones(n),increment) is increment
        assert project(13,np.ones(n),increment) is increment
        for mean in (np.zeros(n),np.ones(n)*1e-20,np.eye(n)[0],np.r_[1.,1.,np.zeros(n-2)]):
            before = mean.tobytes(); assert project(10,mean,increment) is increment
            assert mean.tobytes() == before
        for special in (np.zeros((n,n),np.float32),np.eye(n,dtype=np.float32),increment-increment.T):
            assert np.count_nonzero(project(10,np.ones(n),special)) == 0
    import flopscope as flops
    import flopscope.numpy as fnp
    native = {'fnp': fnp}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), native)
    increment = (np.arange(64,dtype=np.float32).reshape(8,8)%13-6)/np.float32(8)
    mean = 1.0 + np.arange(8)*.03
    expected = project(10,mean,increment)
    with flops.BudgetContext(flop_budget=1000000,quiet=True) as context:
        native_mean = fnp.asarray(mean,dtype=fnp.float64)
        native_increment = fnp.asarray(increment,dtype=fnp.float32)
        native_result = native['_gn_perp_host'](10,native_mean,native_increment)
        got = np.asarray(native_result,dtype=np.float64).copy()
        unchanged_mean = np.asarray(native_mean).copy()
        unchanged_increment = np.asarray(native_increment).copy()
    bill = int(context.summary_dict()['flops_used']); assert 0 < bill < 1000000
    np.testing.assert_allclose(got,expected,rtol=2e-11,atol=2e-12)
    assert unchanged_mean.tobytes()==mean.tobytes() and unchanged_increment.tobytes()==increment.tobytes()
    result = dict(passed=True,primitive_passed=True,source_sha256=digest,rows=rows,native_bill=bill,
                  native_max_error=float(np.max(np.abs(got-expected))),
                  checks=['dense constrained least squares','symmetry','zero diagonal','mean annihilation',
                          'idempotence','host scale invariance','input ownership','inactive identity',
                          'zero/tiny/concentrated mean identity fallback','zero/diagonal/skew increment'],
                  no_model_import=True)
    with (D/'algebra_result.json').open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,max_dense_error=max(x['max_dense_error'] for x in rows))))


if __name__ == '__main__': main()
