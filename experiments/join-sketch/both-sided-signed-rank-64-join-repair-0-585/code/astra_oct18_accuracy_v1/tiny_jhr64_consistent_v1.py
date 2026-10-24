"""Actual public FNP0.12.3 algebra/ownership; no model/runtime qualification."""
import importlib.metadata, json
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import numpy as fnp
assert importlib.metadata.version('flopscope') == '0.12.3'
D=Path(__file__).resolve().parent
rng=np.random.default_rng(20261018)
n,m,k,r,prefix=1024,20,12,8,4
q=np.linalg.qr(rng.normal(size=(m,r)))[0].astype('f')
j=rng.normal(size=(m,k)).astype('f');f=q.T@j;w=rng.normal(size=(m,n)).astype('f')
pp=rng.normal(size=(n,prefix+k+2)).astype('f');pp[:,prefix:prefix+k]=(w.T@q)@f
old=pp.copy();savedq=q.copy();savedf=f.copy()
ip=np.array([0,1,4,5,8,9]);im=ip+2
lam=np.ones(pp.shape[1],dtype='f');lam[prefix+ip]=np.arange(1,7)/6;lam[prefix+im]=-lam[prefix+ip]
response=np.arange(0,n,2);live=np.arange(n-10);outer=np.arange(n-5);tail=np.array([10,11,17])
state=dict(cohorts=[(0,0)]*6+[(prefix,4),(prefix+4,4),(prefix+8,4)])
fa=fnp.asarray
oldf=fa(old);lf=fa(lam);fp=fa(pp)
def parent_read(left,young_right,lam,live,state):
    return fnp.matmul(left*lam[None,:],oldf[live,:].T)
class Base:
    def predict(self,*args):raise ValueError('error path')
env=dict(fnp=fnp,_shared_old384_mm=fnp.matmul,_JOIN_SCHEDULE={13:(6,8)},
         _ASTRA_BCC_CTX=[dict(outer_rows=fa(outer))],_OTT_DATA=dict(full_tail=fa(tail)),
         _cp_read_young_sq=parent_read,_RS_LAYER=[13],_RS={13:(1.,1.)},Estimator=type('Est',(Base,),{}))
exec((D/'fragment_join_h64_consistent_v1.py').read_text(),env)
with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
    env['_jhr_prepare'](fa(w),fa(np.arange(m)),fa(j),fa(q),fa(f),prefix,prefix+k,lf[prefix:prefix+k]**2,state)
    got=env['_jhr_apply'](fp)
    assert got is fp
    corrected=np.asarray(got).copy()
    # A nontrivial pair-law left fold is recomputed from corrected entries.
    swap=np.arange(pp.shape[1]);swap[prefix+ip]=prefix+im;swap[prefix+im]=prefix+ip
    fm=fa(np.linspace(.01,.1,n,dtype='f'))
    left=(got*got-fm[:,None]*(.3*got+.7*got[:,swap]))[response,:]
    observed=env['_cp_read_young_sq'](left,None,lf,fa(live),state)
    expected=(np.asarray(left)*lam)@corrected[live,:].T
    np.testing.assert_allclose(np.asarray(observed),expected,rtol=5e-4,atol=.005)
    np.testing.assert_array_equal(corrected[:,:prefix],old[:,:prefix])
    np.testing.assert_array_equal(corrected[:,prefix+k:],old[:,prefix+k:])
    excluded=np.r_[tail,np.arange(n-5,n)]
    np.testing.assert_array_equal(corrected[excluded],old[excluded])
    np.testing.assert_array_equal(q,savedq);np.testing.assert_array_equal(f,savedf)
    # Current-layer scalar K3 uses corrected pp; it must not be frozen old K3.
    k3=np.sum(corrected**3*lam,axis=1)
    k3old=np.sum(old**3*lam,axis=1)
    assert np.max(np.abs(k3-k3old))>1e-4
    assert env['_JHR_CONTEXT'][0] is None
    env['_JHR_CONTEXT'][0]={'sentinel':True}
    try:env['Estimator']().predict(None,None)
    except ValueError:pass
    else:raise AssertionError('exception was swallowed')
    assert env['_JHR_CONTEXT'][0] is None
    res=dict(status='PASS public FNP0.12.3 tiny both-sided CP with recomputed synthetic NP; not full estimator',
             flopscope=importlib.metadata.version('flopscope'),bill_tiny=int(budget.flops_used),
             corrected_read_matches_dense=True,young_prefix_tail_deadcut_unchanged=True,
             Q_F_unchanged=True,scalar_K3_changes_consistently=True,exception_clears_context=True)
p=D/'TINY_JHR64_CONSISTENT_v1.json';assert not p.exists();p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
