import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import json, textwrap
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent
parent = (D.parent/'opus_fix336687_v1/candidate_v61j.py').read_text()
candidate = (D/'fragment_bookkeeping_algebra_v1.py').read_text()
def extract(source, begin):
    start = source.index(begin, source.index('def _astra_fhm_core'))
    end = source.index('                m = m + dm\n',start)
    code = textwrap.dedent(source[start:end])
    ns = {'fnp':fnp, '_ps32': lambda x: x if getattr(x,'dtype','') == 'float32' else fnp.asarray(x,dtype=fnp.float32)}
    fcode = 'def run(C,m,dm,rc):\n    l=1\n    _GN_RC={1:rc}\n    _GN_KC={}\n    _E2_LAYER={}\n    _FULL_LOWER_MOMENTS=True\n' + textwrap.indent(code,'    ') + '    return C,_ps_Cdiag,_ps_post_cov32\n'
    exec(fcode,ns)
    return ns['run']
old = extract(parent, '                _gn_C0 = C\n')
new = extract(candidate, '                # Real-arithmetic identity')
rng=np.random.default_rng(336867)
rows=[]
for n,rc in ((64,0.),(64,-.6539862186840672),(1024,-.557104209804189)):
    C0=np.eye(n)*.4+rng.normal(scale=.00001,size=(n,n))
    m0=rng.uniform(.02,.7,n);dm0=rng.normal(scale=.005,size=n)
    with fs.BudgetContext(flop_budget=2**50,quiet=True):
        C,m,dm=fnp.asarray(C0),fnp.asarray(m0),fnp.asarray(dm0)
    vals=[]
    for f in (old,new):
        with fs.BudgetContext(flop_budget=2**50,quiet=True) as b:
            out,diag,out32=f(C,m,dm,rc)
        vals.append((np.asarray(out).copy(),np.asarray(diag).copy(),int(b.flops_used),len(b._op_log)))
        assert np.array_equal(np.asarray(C),C0)
        assert np.array_equal(np.asarray(m),m0)
        assert np.array_equal(np.asarray(dm),dm0)
    ref=C0-(1+rc)*(np.outer(m0,dm0)+np.outer(dm0,m0)+np.outer(dm0,dm0))
    diagerr=float(np.max(np.abs(vals[1][1]-np.diag(ref))))
    err=float(np.max(np.abs(vals[1][0]-ref)))
    diff=float(np.max(np.abs(vals[1][0]-vals[0][0])))
    assert diagerr<2e-15 and err<2e-8
    rows.append(dict(n=n,rc=rc,diagonal_max_error=diagerr,matrix_max_error=err,parent_candidate_max_difference=diff,parent_flops=vals[0][2],candidate_flops=vals[1][2],flops_saved=vals[0][2]-vals[1][2],charged_operations_saved=vals[0][3]-vals[1][3],inputs_unchanged=True))
# Diagonal real identities: actual range, float64 first then final float32.
x=rng.normal(size=1024);y=rng.normal(size=1024);z=rng.normal(size=1024)
identity_errors={
    'trace':float(np.max(np.abs((x*y+x*(2*y))-x*(3*y)))),
    'k22':float(np.max(np.abs((.3*(x*x+2*x*x))/3-.3*x*x))),
    'star':float(np.max(np.abs((3*x-2*x).astype('f4')-x.astype('f4')))),
}
receipt=dict(passed=True,flopscope=fs.__version__,scope='Actual public-FNP mean-covariance code extracted from both sources; no model, rounding efficacy untested.',rows=rows,diagonal_identity_errors=identity_errors)
(D/'TINY_BOOKKEEPING_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
