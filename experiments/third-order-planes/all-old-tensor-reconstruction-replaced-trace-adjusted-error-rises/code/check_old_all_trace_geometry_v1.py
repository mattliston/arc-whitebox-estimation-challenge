"""Gaussian fourth-moment projection witness; tiny arrays, no endpoint targets.

The 6^3 equiprobable quadrature rows match standard Gaussian moments through
degree four. The residual cubic must be orthogonal to every linear coordinate.
Both materialized and orthogonally mapped lazy factor representations are used.
This certifies the trace projection identity, not endpoint accuracy.
"""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
base = HERE/'candidate_old_both_tail_trace_bounded_v6.py'
source = base.read_text()
assert hashlib.sha256(base.read_bytes()).hexdigest() == '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
trace_node = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_ott_trace'][-1]
trace = ast.get_source_segment(source,trace_node)
ns = dict(fnp=np, _OTT_CONTEXT=[None], _OTT_DATA={}, _OTT_STATS=dict(calls=0,rows_omitted=0),
          _shared_old384_mm=np.matmul, _shared_old384_finite=lambda a,label:None)
exec(trace,ns)
exec((HERE/'old_all_trace_research_fragment_v1.py').read_text(),ns)
nodes = [-np.sqrt(3),0,0,0,0,np.sqrt(3)]
U = np.array(list(itertools.product(nodes,repeat=3)),dtype=np.float64)
rng = np.random.default_rng(20260926)
F = rng.normal(size=(3,18))
lam = np.concatenate([np.ones(9),-np.ones(9)])
Q = np.linalg.qr(rng.normal(size=(3,3)))[0]
records = []
for lazy in (False,True):
    state = dict(columns=18,F=F.copy(),lazy=None)
    fullF = F
    if lazy:
        state['lazy'] = dict(map=Q,base=F[:,:12].copy(),join=F[:,12:].copy(),prefix=12)
        fullF = np.concatenate([Q@F[:,:12],F[:,12:]],axis=1)
    before = state['F'].copy()
    ns['_OTT_DATA'].clear()
    ns['_OTT_CONTEXT'][0] = dict(lam=lam,rows=None,n=len(U))
    output = ns['_slr_reconstruct'](U,state)
    diagonal = ((U@fullF)**3)@lam
    trace_diagonal = ns['_OTT_DATA']['repair']
    residual_projection = U.T@(diagonal-trace_diagonal)/len(U)
    error = float(np.max(np.abs(residual_projection)))
    assert error < 1e-11
    assert output.shape == (len(U),18) and np.count_nonzero(output)==0
    assert np.array_equal(state['F'],before)
    records.append(dict(lazy=lazy,max_residual_linear_moment=error,carrier_unchanged=True))
result = dict(scope=__doc__,passed=True,records=records,endpoint_validation='not run')
with (HERE/'old_all_trace_geometry_v1.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
