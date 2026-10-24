"""Independent entrywise and quadratic-form symmetry certificates."""
import hashlib
import json
from pathlib import Path
import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((HERE/'build.json').read_text())
assert sha(HERE/'symmetry_fnp.py') == build['fragment_sha256']
assert sha(HERE/'candidate.py') == build['candidate_sha256']
namespace = {'fnp':fnp}
exec(compile((HERE/'symmetry_fnp.py').read_text(),str(HERE/'symmetry_fnp.py'),'exec'),namespace)
project = namespace['_covsym_full']
rng = np.random.default_rng(2026092001)
a = rng.normal(size=(17,17))
b = rng.normal(size=(17,17))
sym = a@a.T
fixtures = [('symmetric_psd',sym), ('indefinite',np.diag(np.linspace(-2.,3.,17))),
            ('skewed_covariance',sym+1e-3*(b-b.T)),
            ('general',a), ('pure_skew',b-b.T),
            ('diagonal',np.diag(np.linspace(.01,4.,17))), ('zero',np.zeros((17,17)))]
directions = rng.normal(size=(17,7))

def run():
    arrays, rows = [], []
    with flops.BudgetContext(flop_budget=10**7,quiet=True) as ctx:
        for name,x in fixtures:
            xx = fnp.asarray(x)
            before = np.asarray(xx).tobytes()
            yy = project(xx)
            yy2 = project(yy)
            y = np.asarray(yy).copy()
            target = np.empty_like(x)
            for i in range(len(x)):
                for j in range(len(x)):
                    target[i,j] = (float(x[i,j])+float(x[j,i]))/2
            assert np.array_equal(y,target)
            assert np.array_equal(y,y.T) and np.array_equal(y,np.asarray(yy2))
            assert np.diag(y).tobytes() == np.diag(x).tobytes()
            assert np.asarray(xx).tobytes() == before
            old_q = np.sum(directions*(x@directions),axis=0)
            new_q = np.sum(directions*(y@directions),axis=0)
            error = float(np.max(np.abs(old_q-new_q))/max(1.,float(np.max(np.abs(old_q)))))
            assert error <= 1e-12
            if np.array_equal(x,x.T):
                assert y.tobytes() == x.tobytes()
            rows.append(dict(case=name,quadratic_relative_scaled_error=error,
                             explicit_entries_exact=True,diagonal_bytes_exact=True,
                             symmetry_exact=True,idempotent=True,input_bytes_unchanged=True))
            arrays.append(y)
    return arrays,rows,ctx.summary_dict()['flops_used']

arrays,rows,bill = run()
again,rows2,bill2 = run()
assert rows == rows2 and bill == bill2
assert all(x.tobytes()==y.tobytes() for x,y in zip(arrays,again))
rejections=[]
with flops.BudgetContext(flop_budget=10**6,quiet=True) as ctx:
    for name,x in [('nonsquare',np.ones((2,3))),('nonmatrix',np.ones(3)),
                   ('wrong_dtype',np.ones((3,3),np.float32))]:
        try:
            project(fnp.asarray(x))
        except ValueError:
            rejections.append(name)
        else:
            raise AssertionError(name)
out = HERE/'unit_v1'
out.mkdir(exist_ok=False)
record = dict(passed=True,cases=rows,repeated_arrays_byte_exact=True,
              repeated_bills_exact=True,bill_per_batch=bill,rejections=rejections,
              source_sha256=sha(__file__),candidate_sha256=build['candidate_sha256'],
              max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'result.json').open('x') as stream:
    json.dump(record,stream,indent=2)
print(json.dumps(record))
