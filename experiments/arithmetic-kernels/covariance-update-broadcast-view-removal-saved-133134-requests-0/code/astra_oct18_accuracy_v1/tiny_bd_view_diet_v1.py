import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import ast
import json
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent
parent = (D.parent/'astra_oct18_bill_v1/candidate_v61j_bookkeeping_union_HELD_v1.py').read_text()
fragment = (D/'fragment_bd_view_diet_v1.py').read_text()
assert fs.__version__.split('+')[0] == '0.12.3', fs.__version__

class Hold(list):
    def __call__(self, x):
        self.append(x)
        return x

def extract(source, name):
    node = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
    return ast.get_source_segment(source, node)

def namespace(changed):
    ns = dict(fnp=fnp, _E2Hold=Hold, _BD_J1_PHYS=.7225448789078501,
              _S19_BD_ADD=(.024419009444819114, 0., .3928502699992539, .15153807567151212),
              _BD32_STATE={}, _E2_LAYER={}, _G5_CONST={}, _IDC={},
              _IDC_STATS=dict(hits=0,misses=0), _ps_k2132=lambda x:x, _ps_pre32=lambda x:x)
    for name in ('_bd32_state','_g5_arr','_g5_j1','_g5_sub','_g5_chain','_g5_mulrow','_idc_get'):
        exec(extract(parent,name),ns)
    if changed:
        exec(extract(fragment,'_o18_bd_literal_column'),ns)
    exec(extract(fragment if changed else parent,'_bd_update'),ns)
    return ns

rng = np.random.default_rng(336867)
rows = []
for ndim in (1,2):
  for pattern in ('finite','zeros','mixed'):
    n=8
    arrays = dict(bp=rng.normal(size=(n,n)).astype('f4'),
                  C=rng.normal(size=(n,n)).astype('f4'), K=rng.normal(size=(n,n)).astype('f4'),
                  d=rng.normal(size=n), a=rng.normal(size=n), t=rng.normal(size=n), s=rng.normal(size=n),
                  alpha=rng.normal(size=n), sigma=rng.uniform(.2,1.5,n), gate=rng.uniform(0,1,n),
                  alive=rng.random(n)>.3, births=rng.random(n)>.4,
                  coefficients=rng.normal(size=(4,) if ndim==1 else (4,n)).astype('f4'))
    if pattern != 'finite':
        for name in ('bp','C','K','d','a','t','s','alpha','gate','coefficients'):
            flat=arrays[name].reshape(-1)
            if pattern=='zeros':
                flat[:]=0
            flat[::3]=np.float32(-0.)
            flat[1::5]=np.float32(0.)
    with fs.BudgetContext(flop_budget=2**50,quiet=True):
        inp={k:fnp.asarray(v) for k,v in arrays.items()}
    aux={k:inp[k] for k in ('alpha','sigma','gate')}
    values=[]
    for changed in (False,True):
        ns=namespace(changed)
        with fs.BudgetContext(flop_budget=2**50,quiet=True):
            ns['_bd32_state'](n)
            ns['_g5_j1']((1.,0.,1.),ns['_BD_J1_PHYS'])
            ns['_g5_arr'](ns['_S19_BD_ADD'][:3])
            if changed:
                ns['_o18_bd_literal_column'](ns['_S19_BD_ADD'][:3])
        owners = (ns['_BD32_STATE'][n]['D'], ns['_BD32_STATE'][n]['old32'])
        with fs.BudgetContext(flop_budget=2**50,quiet=True) as budget:
            new,delta=ns['_bd_update'](inp['bp'],inp['C'],inp['K'],aux,inp['d'],inp['alive'],
                                      inp['births'],inp['a'],inp['t'],inp['s'],inp['coefficients'])
        st=ns['_BD32_STATE'][n]
        got=[np.asarray(x).copy() for x in (new,delta,st['D'],st['old32'])]
        values.append((got,int(budget.flops_used),len(budget._op_log)))
        assert owners[0] is st['D'] and owners[1] is st['old32']
        assert str(new.dtype)=='float32' and str(delta.dtype)=='float32'
        assert not np.shares_memory(np.asarray(new),np.asarray(st['old32']))
        assert not np.shares_memory(np.asarray(new),np.asarray(st['D']))
        for k,v in arrays.items():
            assert np.asarray(inp[k]).tobytes()==v.tobytes(), (k,changed)
    tests=[]
    for label,x,y in zip(('new32','delta','D','old32'),values[0][0],values[1][0]):
        tests.append(dict(label=label,byte_exact=x.tobytes()==y.tobytes(),
                          max_difference=float(np.max(np.abs(x-y))),
                          differing_signbits=int(np.count_nonzero(np.signbit(x)!=np.signbit(y)))))
    rows.append(dict(ndim=ndim,pattern=pattern,tests=tests,parent_flops=values[0][1],candidate_flops=values[1][1],
                     extra_flops=values[1][1]-values[0][1],expected_extra_flops=7*n*n,
                     parent_charged_ops=values[0][2],candidate_charged_ops=values[1][2],
                     persistent_owners_preserved=True,new32_owned=True,inputs_unchanged=True))
passed=all(t['byte_exact'] for row in rows for t in row['tests']) and all(row['extra_flops']==row['expected_extra_flops'] for row in rows)
receipt=dict(passed=passed,flopscope=fs.__version__,scope='Extracted complete active bd_update; six fixed tiny cases; public SDK; no model.',rows=rows)
(D/'TINY_BD_VIEW_DIET_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
raise SystemExit(0 if passed else 1)
