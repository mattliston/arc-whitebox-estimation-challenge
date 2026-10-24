"""Small algebra checks for full/compact rows and materialized/lazy factors."""
import itertools
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ns = dict(fnp=np, _slr_finish_factors=lambda state,factors:None,
          _OTT_CONTEXT=[None], _OTT_DATA={}, _OTT_STATS=dict(calls=0,rows_omitted=0),
          _OTT_Z=2.5, _OTT_NEG_Z=2.5,
          _shared_old384_mm=np.matmul, _shared_old384_finite=lambda value,label:None)
exec((HERE/'old_local_rank_trace_fragment_v1.py').read_text(),ns)
ns['_LRR_RANK'] = 2
rng = np.random.default_rng(20260926)
U = np.array(list(itertools.product([-np.sqrt(3),0,0,0,0,np.sqrt(3)],repeat=3)))
F = rng.normal(size=(3,18))
Q = np.linalg.qr(rng.normal(size=(3,3)))[0]
lam = np.r_[np.ones(9), -np.ones(9)]
alpha = np.linspace(-4,4,len(U))
live = alpha > -1
score = np.where((alpha < -2.5)&(~live), alpha+2.5, 2.5-alpha)
kept = min(len(U), ((score>=0).sum()+63)//64*64)
rows = np.argsort(-score,kind='stable')[:kept]
records = []
for lazy in (False,True):
    for compact in (False,True):
        state = dict(columns=18,F=F.copy(),lazy=None)
        canonical = F.copy()
        if lazy:
            state['F'] = None
            state['lazy'] = dict(map=Q,base=F[:,:12].copy(),join=F[:,12:].copy(),prefix=12)
            canonical = np.c_[Q@F[:,:12],F[:,12:]]
        rowmap = rng.permutation(len(U))+7 if compact else None
        n = len(U)+14 if compact else len(U)
        full = np.arange(len(U)) if rowmap is None else rowmap
        ns['_OTT_DATA'].clear()
        ns['_OTT_CONTEXT'][0] = dict(lam=lam,rows=rowmap,n=n,a=alpha,live=live)
        local = ns['_slr_reconstruct'](U,state)
        expected = np.zeros_like(local)
        expected[rows] = U[rows,:2]@canonical[:2]
        assert np.allclose(local,expected,rtol=0,atol=1e-12)
        rho = np.mean(U*U)
        trace_full = 3*rho*U@(canonical@(lam*np.sum(canonical*canonical,axis=0)))
        trace_short = 3*rho*U[:,:2]@(canonical[:2]@(lam*np.sum(canonical[:2]**2,axis=0)))
        repair = np.zeros(n)
        repair[full] = trace_full
        repair[full[rows]] -= trace_short[rows]
        diag_error = float(np.max(np.abs(repair-ns['_OTT_DATA']['repair'])))
        assert diag_error < 1e-11
        a,beta = rng.normal(size=n),rng.normal(size=n)
        de,ds = ns['_lrr_radial_delta'](a,beta)
        exact_ds = beta[full]@(U@canonical-local)
        exact_de = (np.sum(canonical**2,axis=0)*np.sum(a[full]*np.mean(U**2,axis=1))
                    - np.sum(canonical[:2]**2,axis=0)*np.sum(a[full[rows]]*np.mean(U[rows,:2]**2,axis=1)))
        linear_error = float(np.max(np.abs(ds-exact_ds)))
        energy_error = float(np.max(np.abs(de-exact_de)))
        assert linear_error < 1e-10 and energy_error < 1e-10
        if lazy:
            cached = state['local_read_factors']
            assert ns['_lrr_factors'](state) is cached
            assert np.array_equal(state['lazy']['base'],F[:,:12])
        else:
            assert np.array_equal(state['F'],F)
        ns['_slr_finish_factors'](state,None)
        assert state['local_read_factors'] is None
        records.append(dict(lazy=lazy,compact=compact,diagonal_error=diag_error,
                            radial_linear_error=linear_error,radial_surrogate_energy_error=energy_error,
                            full_carrier_unchanged=True,cache_released_at_join=True))
result = dict(passed=True,scope=__doc__,records=records,
              limitation='Algebra of the stated approximation only; no endpoint accuracy or bill claim.')
with (HERE/'old_local_rank_geometry_v1.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
