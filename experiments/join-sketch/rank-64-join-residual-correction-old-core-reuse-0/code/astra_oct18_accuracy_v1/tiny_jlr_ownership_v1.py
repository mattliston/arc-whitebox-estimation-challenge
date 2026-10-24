"""Public-FNP tiny test of new hook and actual current UV split inner read."""
import ast, json, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import numpy as fnp
D=Path(__file__).resolve().parent
P=D.parent/'opus_fix336687_v1/candidate_v61j.py'
s=P.read_text(); tree=ast.parse(s)
node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_uvp_inner'][-1]
uvsource=ast.get_source_segment(s,node)
class Hold:
    def __init__(self):self.items=[]
    def __call__(self,x):self.items.append(x);return x
    def clear(self):self.items.clear()
class Base:
    def predict(self,*a):raise ValueError('test error propagation')
rng=np.random.default_rng(20261018)
out=[]
for mode in ('exact256','residual64'):
    n,m,K,r,ru=18,20,12,8,5
    response=np.array([1,3,4,6,8,9,10,12,14,16])
    previous=np.arange(m)
    j=rng.normal(size=(m,K)).astype('f')
    q,_=np.linalg.qr(rng.normal(size=(m,r)).astype('f'))
    f=q.T@j; w=rng.normal(size=(m,n)).astype('f')
    prefix=4; total=prefix+K+2
    pp=rng.normal(size=(n,total)).astype('f'); pp[:,prefix:prefix+K]=(w.T@q)@f
    pp2=pp*pp; npterm=rng.normal(size=(len(response),total)).astype('f')*.01
    saved=[x.copy() for x in (pp,pp2,npterm,j,f)]
    fa=lambda x:fnp.asarray(x)
    a=fa(np.linspace(-2,2,n,dtype='f')); sig=fa(np.linspace(.8,1.2,n,dtype='f'))
    npp=fa(npterm); fp=fa(pp); fp2=fa(pp2)
    def parent(l,pp2,pp,response,*args):return pp2[response,:]+npp
    env=dict(fnp=fnp,_np_left=parent,Estimator=type('Est',(Base,),{}),_E2_LAYER=dict(ecut_rows=fa(response),a=a,s=sig),_shared_old384_mm=fnp.matmul,_E2Hold=Hold,_BH_HOLD=types.SimpleNamespace(append=lambda x:None,clear=lambda:None),_ubx_product=lambda a,b,*args:fnp.matmul(a,b))
    exec(uvsource,env)
    exec((D/'fragment_join_left_repair_v1.py').read_text().replace("_JLR_MODE = 'exact256'",f"_JLR_MODE = {mode!r}",1),env)
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        env['_jlr_prepare'](fa(w),fa(previous),fa(j),fa(q),fa(f),prefix,prefix+K,fa(np.ones(K,dtype='f')))
        c=env['_JLR_CONTEXT'][0]; positions,chosen,st,sp,actual,relative=c
        actual_np=np.asarray(actual).copy(); chosen_np=np.asarray(chosen); pos=np.arange(len(response)) if positions is None else np.asarray(positions)
        left=env['_np_left'](13,fp2,fp,fa(response),None,None,None,None,None,None,None)
        expected=pp2[response]+npterm
        repaired=pp[chosen_np,st:sp]+actual_np if relative else actual_np
        expected[pos,st:sp]+=repaired*repaired-pp2[chosen_np,st:sp]
        np.testing.assert_allclose(np.asarray(left),expected,rtol=1e-6,atol=1e-5)
        # Production's final UV inner path, not an invented alternate read.
        # One cohort split of total even columns; common component truncated to ru.
        pairs=total//2;fu=rng.normal(size=(ru,pairs)).astype('f');fv=rng.normal(size=(r,pairs)).astype('f')
        full=np.zeros((r,pairs),dtype='f');full[:ru]=fu
        factors=np.concatenate((full+fv,full-fv),axis=1)
        lam=rng.normal(size=total).astype('f')
        uv=(fa(fu),fa(fv),[(0,pairs)],ru)
        got=env['_uvp_inner'](left,fa(lam),uv)
        dense=(expected*lam)@factors.T
        np.testing.assert_allclose(np.asarray(got),dense,rtol=2e-5,atol=1e-4)
        # Production later overwrites every CP diagonal from unchanged pp^3 lam.
        before=np.sum(pp*pp*pp*lam,axis=1); after=np.sum(np.asarray(fp)*np.asarray(fp2)*lam,axis=1)
        np.testing.assert_array_equal(before,after)
        for got0,want in zip((np.asarray(fp),np.asarray(fp2),np.asarray(npp),j,f),saved):np.testing.assert_array_equal(got0,want)
        assert env['_JLR_CONTEXT'][0] is None
        # Error path clears ownership too.
        env['_JLR_CONTEXT'][0]=c
        try:env['Estimator']().predict(None,None)
        except ValueError:pass
        else:raise AssertionError('parent error did not propagate')
        assert env['_JLR_CONTEXT'][0] is None
        out.append(dict(mode=mode,pass_all=True,bill=int(ctx.flops_used),read_path='_uvp_inner copied from currentv61j',original_inputs_byte_identical=True,diagonal_byte_identical=True,NP_terms_unmodified=True,exception_context_cleared=True))
res=dict(status='PASS tiny public-FNP algebra/ownership only, not full estimator or native runtime qualification',rows=out)
p=D/'TINY_JLR_OWNERSHIP_v1.json';assert not p.exists();p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
