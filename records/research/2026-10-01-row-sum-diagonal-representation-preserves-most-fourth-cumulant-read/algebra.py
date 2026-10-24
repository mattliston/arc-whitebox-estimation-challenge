"""Root-only tiny witness for the actual refined row/diagonal helper fragment."""
from pathlib import Path
import ast, hashlib, itertools, json

D = Path(__file__).resolve().parent


def main():
    import numpy as np
    source = (D/'fragment.py').read_text()
    names = {'_k4r_residual', '_k4r_project', '_k4r_state', '_k4r_delta'}
    nodes = [x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert {x.name for x in nodes} == names
    env = {'fnp': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual_helpers', 'exec'), env)
    project, delta = env['_k4r_project'], env['_k4r_delta']
    cases = []
    for n in (3, 5, 8):
        z = np.arange(n*n, dtype=np.float64).reshape(n,n)
        arbitrary = ((z*7 % 19)-9); arbitrary = (arbitrary+arbitrary.T)/2
        a = (np.arange(n, dtype=np.float64)*3 % 7)-3
        matrices = [('general', arbitrary), ('diagonal', np.diag(a)), ('constant', np.full((n,n),.75)),
                    ('additive', a[:,None]+a[None,:]+.75+np.diag(a*a-2))]
        w = ((np.arange(n*n).reshape(n,n)*5 % 13)-6).astype(np.float32)
        for label, M in matrices:
            frozen = M.tobytes(), w.tobytes()
            u,v,c = state = project(M)
            hat = u[:,None]+u[None,:]+c+np.diag(v)
            np.testing.assert_allclose(hat.sum(1), M.sum(1), atol=1e-11, rtol=1e-12)
            np.testing.assert_allclose(np.diag(hat), np.diag(M), atol=1e-11, rtol=1e-12)
            residual = M-hat
            np.testing.assert_allclose(residual.sum(1), 0, atol=1e-11)
            np.testing.assert_allclose(np.diag(residual), 0, atol=1e-11)
            if label != 'general': np.testing.assert_allclose(hat, M, atol=1e-11)
            q = w.astype(np.float64)**2
            actual = delta(w, state); matrix = np.sum(q*(hat@q), axis=0)
            centered = q-q.mean(0)
            np.testing.assert_allclose(np.sum(q*(residual@q),axis=0),
                                       np.sum(centered*(residual@centered),axis=0),atol=1e-8,rtol=1e-11)
            tensor = np.zeros((n,n,n,n), dtype=np.float64)
            for i in range(n):
                tensor[i,i,i,i] = hat[i,i]
                for j in range(i+1,n):
                    for ix in set(itertools.permutations((i,i,j,j))): tensor[ix] = hat[i,j]/3
            direct = np.einsum('ijkl,ia,ja,ka,la->a', tensor, w, w, w, w, optimize=True)
            np.testing.assert_allclose(actual, matrix, atol=1e-8, rtol=1e-11)
            np.testing.assert_allclose(actual, direct, atol=1e-8, rtol=1e-11)
            anti = z-z.T
            mixed = project(M+anti)
            for x,y in zip(state,mixed): np.testing.assert_allclose(x,y,atol=1e-11)
            assert frozen == (M.tobytes(),w.tobytes())
            cases.append(dict(n=n, case=label, matrix_error=float(np.max(np.abs(actual-matrix))),
                              fourth_tensor_error=float(np.max(np.abs(actual-direct)))))
        C = np.eye(n)*2 + np.ones((n,n))*.125; radial = .25; cd = np.diag(C)
        post22 = radial/3*(np.outer(cd,cd)+2*C*C); pk4 = radial*cd*cd
        np.fill_diagonal(post22, pk4)
        M = env['_k4r_residual'](pk4,post22,C,radial)
        np.testing.assert_allclose(M,0,atol=1e-12)
        for x in env['_k4r_state'](pk4,post22,C,radial): np.testing.assert_allclose(x,0,atol=1e-12)
    result = dict(passed=True,fragment_sha256=hashlib.sha256(source.encode()).hexdigest(),cases=cases,
                  exact_rows_and_diagonal=True,pure_diagonal_and_additive_exact=True,
                  discarded_symmetric_residual_has_zero_rows_and_diagonal=True,
                  explicit_fourth_tensor_checked=True,inputs_unchanged=True,no_model_job=True)
    with (D/'algebra_result.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
