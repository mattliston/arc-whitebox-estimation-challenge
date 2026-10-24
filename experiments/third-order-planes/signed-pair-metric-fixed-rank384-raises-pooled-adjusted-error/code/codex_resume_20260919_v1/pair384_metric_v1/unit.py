"""Explicit third tensors independently certify the archived paired metric."""
import hashlib
import json
from pathlib import Path
import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((HERE / 'build.json').read_text())
assert sha(HERE / 'metric_fnp.py') == build['fragment_sha256']
assert sha(HERE / 'candidate.py') == build['candidate_sha256']
namespace = {'fnp': fnp}
exec(compile((HERE / 'metric_fnp.py').read_text(), str(HERE / 'metric_fnp.py'), 'exec'), namespace)
metric = namespace['_pair256_metric']
apply_kernel = namespace['_pair256_apply_kernel']
coordinates = namespace['_pair256_coordinates']
rng = np.random.default_rng(20092026)
fixtures = []
for kind, n, pairs, rank, separation in [
    ('ordinary', 8, 5, 3, .7), ('signed', 19, 9, 7, .3),
    ('near_coalesced', 13, 6, 4, 1e-4), ('coalesced', 12, 4, 5, 0.),
    ('zero_coefficients', 9, 7, 4, .2)]:
    base = rng.normal(size=(n, pairs))
    delta = separation*rng.normal(size=(n, pairs))
    j = np.concatenate([base+delta, base-delta], axis=1).astype(np.float32)
    a = rng.normal(size=pairs).astype(np.float32)
    a[0] = np.float32(-0.)
    if kind == 'zero_coefficients':
        a.fill(0)
    coefficient = np.concatenate([a, -a])
    q = np.linalg.qr(rng.normal(size=(n, rank)))[0].astype(np.float32)
    omega = rng.normal(size=(n, rank)).astype(np.float32)
    fixtures.append((kind, j, coefficient, q, omega))


def run():
    output, rows = [], []
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as ctx:
        for kind, j, coefficient, q, omega in fixtures:
            jj, aa, qq, oo = [fnp.asarray(x) for x in (j, coefficient, q, omega)]
            before = np.asarray(aa).tobytes()
            basis, kernel = metric(jj, aa)
            gram = basis @ apply_kernel(basis.T, kernel)
            ranged = basis @ apply_kernel(basis.T @ oo, kernel)
            physical = qq.T @ jj
            projected_coordinates = coordinates(physical)
            core = projected_coordinates @ apply_kernel(projected_coordinates.T, kernel)
            assert np.asarray(aa).tobytes() == before
            assert np.array_equal(np.asarray(physical), q.T @ j)
            actual = {key:np.asarray(value).copy() for key,value in
                      [('gram', gram), ('range', ranged), ('physical', physical), ('core', core)]}
            n, columns = j.shape
            pairs = columns//2
            expected = np.zeros((n,n), np.float64)
            # Direct original x/y third tensors, independently unfolded.
            for i in range(pairs):
                x, y = j[:,i].astype(np.float64), j[:,pairs+i].astype(np.float64)
                tensor = float(coefficient[i])*(np.einsum('i,j,k->ijk',x,x,x)-
                                                np.einsum('i,j,k->ijk',y,y,y))
                unfolding = tensor.reshape(n,n*n)
                expected += unfolding @ unfolding.T
            norm = np.linalg.norm(expected)
            if norm == 0:
                assert all(np.array_equal(actual[k],np.zeros_like(actual[k])) for k in ['gram','range','core'])
                relative, range_relative = 0., 0.
            else:
                relative = float(np.linalg.norm(actual['gram']-expected)/norm)
                target_range = expected @ omega.astype(np.float64)
                range_relative = float(np.linalg.norm(actual['range']-target_range)/np.linalg.norm(target_range))
                assert relative <= 2e-5 and range_relative <= 2e-5, (kind, relative, range_relative)
            output.append(actual)
            rows.append(dict(kind=kind, gram_relative=relative, range_relative=range_relative,
                             physical_projection_byte_exact=True, coefficient_bytes_unchanged=True,
                             finite=all(np.isfinite(x).all() for x in actual.values())))
    return output, rows, ctx.summary_dict()['flops_used']


first, rows, first_bill = run()
second, repeated_rows, second_bill = run()
assert rows == repeated_rows and first_bill == second_bill
assert all(np.array_equal(a[k], b[k]) for a,b in zip(first,second) for k in a)
rejected = []
with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as reject_ctx:
    j = fnp.asarray(fixtures[0][1])
    a = fixtures[0][2].copy()
    a[-1] += np.float32(.1)
    cases = [
        ('non_opposed', lambda: metric(j,fnp.asarray(a))),
        ('odd_columns', lambda: metric(j[:,:-1],fnp.asarray(fixtures[0][2][:-1]))),
        ('zero_range', lambda: namespace['_pair256_nonzero_range'](fnp.zeros((8,3),dtype=fnp.float32))),
        ('wrong_dtype', lambda: metric(j.astype(fnp.float64),fnp.asarray(fixtures[0][2]))),
    ]
    for name, call in cases:
        try:
            call()
        except RuntimeError:
            rejected.append(name)
        else:
            raise AssertionError(('missing rejection',name))
out = HERE / 'unit_v1'
out.mkdir(exist_ok=False)
result = dict(passed=True, cases=rows, repeated_arrays_byte_exact=True,
              repeated_bills_exact=True, bill_per_batch=first_bill,
              rejected=rejected, rejection_bill=reject_ctx.summary_dict()['flops_used'],
              source_sha256=sha(__file__), fragment_sha256=build['fragment_sha256'],
              candidate_sha256=build['candidate_sha256'],
              max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'result.json').open('x') as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(result))
