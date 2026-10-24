from pathlib import Path
import importlib.util, json, time, types
import torch as t
t.set_num_threads(1)
t.set_num_interop_threads(1)
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('port',D/'torch_rtt_current95c_v1.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
t.manual_seed(313)
start=time.monotonic();results=[]
for old in (0,8):
    n,m,k=96,64,24
    U=t.randn(n,4)*.1; F=t.randn(4,old)*.1
    pp=t.cat((U@F,t.randn(n,k-old)*.1),1).requires_grad_()
    lam=t.randn(k)
    response=t.arange(m);live=t.arange(n)
    vc=t.linspace(.5,1.5,n,dtype=t.float64)
    alpha=t.cat((t.linspace(-.5,.5,40),t.ones(56)*-1.7)).to(t.float64)
    mu=alpha*vc.sqrt()
    forms=t.randn(m,n)*1e-4
    factor=t.randn(m,dtype=t.float64)*.01
    w=t.randn(k,dtype=t.float64)*.01;x=t.randn(k,dtype=t.float64)*.01
    partner=t.arange(k-1,-1,-1)
    # Keep the old cohort's partners inside the old component.
    if old:partner=t.cat((t.arange(old-1,-1,-1),t.arange(k-1,old-1,-1)))
    cd=types.SimpleNamespace(t=t,call_layer=3,ncar6_state={'mu':mu,'C':t.diag(vc)},
        native_push_context=dict(layer=3,old_columns=old,truncated=False,
            mapping=None,ru=4,blocks=[dict(lo=0,hi=old,F=F,pairs=())]),
        shared={},env={})
    cd._cutsim_begin=lambda layer,mean,cov: None
    def parent(tt, pp, lam, response, live, U, F, old_columns):
        ly=pp[response,old_columns:]
        base_y=(ly*ly*lam[old_columns:])@pp[live,old_columns:].T
        if old_columns:
            base_old=((pp[response,:old_columns]**2*lam[:old_columns])@F.T)@U[live].T
            cd._o18_rtt_base_old=(3,base_old,lam)
        else:
            base_old=t.zeros_like(base_y);cd._o18_rtt_base_old=None
        law_left=pp[response].to(t.float64)*w+pp[response][:,partner].to(t.float64)*x
        fold=factor[:,None]*(law_left@pp[live].to(t.float64).T)
        old_fold=factor[:,None]*(law_left[:,:old_columns]@pp[live,:old_columns].to(t.float64).T)
        cd._o18_rtt_law=dict(layer=3,forms=forms.to(t.float64),old_fold=old_fold,
            factor=factor,old_w=w[:old_columns],old_x=x[:old_columns],old_partner=partner[:old_columns])
        return base_y+base_old+fold.to(t.float32)+forms
    cd.shared['read']=parent
    cd.env['features']=lambda layer,*args: t.ones(n,30,dtype=t.float32)*.01
    p.install_after_m(cd,None)
    cd._cutsim_begin(3,mu,t.diag(vc))
    out=cd.shared['read'](t,pp,lam,response,live,U,F,old)
    features=cd.env['features'](3)
    assert cd._o18_rtt_port['pending'] is None
    assert cd._o18_rtt_port['reads'][0]['c15']==64
    assert cd._o18_rtt_port['reads'][0]['c2']==96
    loss=out.square().mean()+features.square().mean()
    loss.backward()
    assert pp.grad is not None and t.isfinite(pp.grad).all() and pp.grad.abs().sum()>0
    # The preserved forms never enter the residual projection.
    results.append(dict(old=old,finite=True,gradient_abs_sum=float(pp.grad.abs().sum()),
        feature22_changed=bool((features[:,22]!=.01).any()),read=cd._o18_rtt_port['reads'][0]))
receipt=dict(status='TINY_FORWARD_GRADIENT_PASS',cases=results,seconds=time.monotonic()-start,
    limitation='Synthetic card/kernel checks only; not full worker replay or native parity.')
(D/'TINY_PORT_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
