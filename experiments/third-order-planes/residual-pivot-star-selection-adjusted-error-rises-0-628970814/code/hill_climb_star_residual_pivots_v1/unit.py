"""Tiny paid native fixtures: extract only the new helper/factor, no estimator."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    assert not (HERE / 'unit_result.json').exists()
    receipt = json.loads((HERE / 'build.json').read_text())
    for key in ('candidate', 'fragment', 'parent', 'protocol'):
        assert receipt[key] == pin(Path(receipt[key]['path']))
    parent = Path(receipt['parent']['path']).read_bytes()
    source = (HERE / 'fragment.py').read_bytes()
    assert (HERE / 'candidate.py').read_bytes() == parent + b'\n\n' + source
    tree = ast.parse(source)
    assert [n.name for n in tree.body if isinstance(n, ast.FunctionDef)] == ['_srp_columns', '_lr_factor']
    env = dict(fnp=fnp, _LR_K=4, _LR_Q=1, _LRN_POWER=True, _LR_R=None, _tsr_mm=lambda a, b: a @ b)
    exec(compile(tree, str(HERE / 'fragment.py'), 'exec'), env)
    arrays = []
    arrays.append(('zero', np.zeros((8, 8), np.float32), [0, 1, 2, 3], 0, 1))
    rank1 = np.zeros((8, 8), np.float32); rank1[0] = [1, 2, 6, 3, 5, 8, 4, 7]
    arrays.append(('rank_one_nonprefix_stable_fill', rank1, [2, 4, 5, 7], 1, 1))
    arrays.append(('diagonal', np.diag(np.arange(8, 0, -1).astype(np.float32)), [0, 1, 2, 3], 3, 0))
    arrays.append(('stable_equal_energy_ties', np.eye(8, dtype=np.float32), [0, 1, 2, 3], 3, 0))
    redundant = np.zeros((8, 8), np.float32)
    redundant[0, :3] = [8, 7, 6]
    redundant[1, 3] = 5; redundant[2, 4] = 4; redundant[3, 5] = 3
    redundant[4, 6] = .5; redundant[5, 7] = .25
    arrays.append(('redundant_high_energy_columns', redundant, [0, 3, 4, 5], 3, 0))
    rows = []
    for name, array, expected, updates, fallbacks in arrays:
        before = dict(env['_SRP_STATS'])
        with flops.BudgetContext(flop_budget=2**28, quiet=True) as budget:
            a = fnp.asarray(array); a64 = a.astype(fnp.float64)
            norms = fnp.sum(a64 * a64, axis=0)
            cols = env['_srp_columns'](a, a64, norms, 4)
            assert np.asarray(cols).tolist() == expected
            assert len(set(np.asarray(cols).tolist())) == 4
            assert np.asarray(a).tobytes() == array.tobytes()
        now = env['_SRP_STATS']
        assert now['updates'] - before['updates'] == updates
        assert now['fallbacks'] - before['fallbacks'] == fallbacks
        assert budget.summary_dict()['flops_used'] > 0
        rows.append(dict(name=name, indices=expected, updates=updates, fallbacks=fallbacks,
                         flops=int(budget.summary_dict()['flops_used']), passed=True))
    # Independently reproduce exactly the unchanged post-selection native path.
    with flops.BudgetContext(flop_budget=2**28, quiet=True) as budget:
        a = fnp.asarray(redundant)
        Q, B = env['_lr_factor'](a)
        fixed = fnp.asarray([0, 3, 4, 5], dtype=fnp.int64)
        refQ = fnp.linalg.qr(a[:, fixed])[0]
        refQ = fnp.linalg.qr(a @ fnp.linalg.qr(a.T @ refQ)[0])[0]
        refB = refQ.T @ a
        assert np.asarray(Q).tobytes() == np.asarray(refQ).tobytes()
        assert np.asarray(B).tobytes() == np.asarray(refB).tobytes()
        np.testing.assert_allclose(np.asarray(Q.T @ Q), np.eye(4), rtol=0, atol=1e-6)
        assert np.asarray(a).tobytes() == redundant.tobytes()
    assert all(not isinstance(v, np.ndarray) for v in env['_SRP_STATS'].values())
    result = dict(passed=True, candidate=receipt['candidate'], fragment=receipt['fragment'],
                  protocol=receipt['protocol'], unit=pin(Path(__file__).resolve()), cases=rows,
                  final_factor_path_byte_exact=True, final_factor_flops=int(budget.summary_dict()['flops_used']),
                  helper_only=True, estimator_imported=False, estimator_setup_called=False, no_targets=True)
    with (HERE / 'unit_result.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
