import ast
import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import *

seal_check()
source = PARENT.read_text()
tree = ast.parse(source)
sel = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'select'][-1]
ns = {'fnp': fnp}
exec(ast.get_source_segment(source, sel), ns)
exec((HERE / 'helper.py').read_text(), ns)
rng = np.random.default_rng(7601024)
selectors = []
for n in [0, 1, 7, 32, 33, 64, 1024]:
    for dtype in [np.float32, np.float64]:
        for kind in range(3):
            x = (rng.uniform(-.02, 1.02, n) if kind == 0 else
                 np.full(n, .5) if kind == 1 else
                 np.resize(np.array([0, .009, .01, .99, 1, .01, -.1]), n)).astype(dtype)
            original = x.tobytes()
            order = np.argsort(-x, kind='stable')
            for enabled in [False, True]:
                ns['_FULL_CUBIC_SUPPORT'] = enabled
                count = n if enabled else min(n, ((int(np.count_nonzero(x >= .01)) + 31) // 32) * 32)
                expected = order[:count]
                expected_mask = np.zeros(n, bool)
                expected_mask[expected] = True
                outputs, bills = [], []
                with flops.BudgetContext(flop_budget=2**41, quiet=True):
                    fx = fnp.asarray(x)
                for _ in range(2):
                    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                        ii, mm = ns['_cubic_support_select'](fx, .01)
                    a, b = np.asarray(ii).copy(), np.asarray(mm).copy()
                    assert np.array_equal(a, expected) and np.array_equal(b, expected_mask)
                    assert np.asarray(fx).tobytes() == original
                    outputs.append((a.tobytes(), b.tobytes()))
                    bills.append(int(ctx.summary_dict()['flops_used']))
                assert outputs[0] == outputs[1] and bills[0] == bills[1] and x.tobytes() == original
                selectors.append(dict(n=n, dtype=np.dtype(dtype).name, fixture=kind,
                    enabled=enabled, count=count, repeat_bytes_and_bill_exact=True, flops=bills[0]))

def relative(actual, reference):
    norm = float(np.linalg.norm(reference))
    err = float(np.linalg.norm(actual - reference))
    return err / norm if norm else err

def dense(P, lam):
    n, r = P.shape
    return np.array([[[sum(float(lam[a])*float(P[i,a])*float(P[j,a])*float(P[k,a])
                          for a in range(r)) for k in range(n)] for j in range(n)] for i in range(n)])

def plane(T):
    n = len(T)
    return np.array([[T[i, i, j] for j in range(n)] for i in range(n)])

def star(A):
    n = len(A)
    return np.array([[[(float(A[k,i]) if i == j else 0.) / 3
                    + (float(A[j,i]) if i == k else 0.) / 3
                    + (float(A[i,j]) if j == k else 0.) / 3
                    for k in range(n)] for j in range(n)] for i in range(n)])

tensor_checks = []
for case in range(6):
    n, r = 5, 6
    P = rng.normal(size=(n, r))
    lam = rng.normal(size=r)
    gate = np.array([0., .005, .5, .999, 1.])
    if case == 1:
        P[:, 1] = P[:, 0]; lam[1] = -lam[0]
    elif case == 2:
        lam[:] = 0
    elif case == 3:
        P = P[:, :0]; lam = lam[:0]
    elif case == 4:
        gate[:] = 1
    elif case == 5:
        gate[:] = 0
    W = rng.normal(size=(n, n)) / 2
    T = dense(P, lam)
    gated = np.einsum('i,j,k,ijk->ijk', gate, gate, gate, T)
    pushed_reference = np.einsum('ai,bj,ck,abc->ijk', W, W, W, gated, optimize=True)
    U = W.T @ (gate[:, None] * P)
    pushed = dense(U, lam)
    read_plane = (U*U) @ (U*lam).T
    read_diagonal = np.sum((U*U*U)*lam, axis=1)
    target = rng.normal(size=(n, n))
    rest = target - read_plane
    A = 3*rest.T - 2*np.diag(np.diag(rest))
    restored = plane(pushed + star(A))
    errors = dict(transport=relative(pushed, pushed_reference),
        repeated_plane=relative(read_plane, plane(pushed_reference)),
        diagonal=relative(read_diagonal, np.array([pushed_reference[i,i,i] for i in range(n)])),
        star_restore=relative(restored, target))
    assert max(errors.values()) <= 2e-12, (case, errors)
    tensor_checks.append(dict(case=case, errors=errors))

# A generic de-repeated tensor witness, not a claim about a padded toy selector.
n = 5
P = np.ones((n, 1)); lam = np.ones(1)
T = dense(P, lam)
old_plane = plane(T)
A = -3*old_plane.T + 2*np.diag(np.diag(old_plane))
T = T + star(A)
gate = np.array([.005, .4, .7, .9, 1.])
masked_gate = gate.copy(); masked_gate[0] = 0
full = np.einsum('i,j,k,ijk->ijk', gate, gate, gate, T)
masked = np.einsum('i,j,k,ijk->ijk', masked_gate, masked_gate, masked_gate, T)
assert np.array_equal(plane(full), np.zeros((n, n)))
assert np.array_equal(plane(masked), np.zeros((n, n)))
ray = np.ones(n)
future_gap = float(np.einsum('i,j,k,ijk->', ray, ray, ray, full-masked))
assert abs(future_gap) > 1e-5
write(HERE / 'unit.json', dict(passed=True, selectors=selectors, tensor_checks=tensor_checks,
    selector_cases=len(selectors), dense_cases=len(tensor_checks),
    strict_tensor_max_error=max(v for r in tensor_checks for v in r['errors'].values()),
    low_gate_witness=dict(current_repeated_planes_both_zero=True, future_cubic_gap=future_gap,
        scope='generic de-repeated tensor and explicit mask; not a padded small native roster'),
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True, selector_cases=len(selectors), dense_cases=len(tensor_checks)), flush=True)

