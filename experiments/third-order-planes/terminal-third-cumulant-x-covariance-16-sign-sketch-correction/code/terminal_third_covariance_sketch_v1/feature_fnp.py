"""Fixed terminal T-times-C sketch; arithmetic entirely in FlopScope."""
import flopscope.numpy as fnp
from probes import SIGNS

def probes():return fnp.asarray(SIGNS,dtype=fnp.float32)

def prepare(p,lam,w,A,alive_prev,B,b,C,g,pdf,sigma,alive,Cpost,K21):
    a=A.astype(fnp.float32)*(alive_prev[:,None]*alive_prev[None,:])
    fnp.fill_diagonal(a,fnp.diag(A))
    c=C.astype(fnp.float32);c=(c+c.T)*.5
    bm=B.astype(fnp.float32);bm=(bm+bm.T)*.5
    d=pdf/sigma
    return dict(p=p,lam=lam,w=w,A=a,B=bm,b=b.astype(fnp.float32),C=c,
                g=g.astype(fnp.float32)*alive,d=d.astype(fnp.float32)*alive,
                Cpost=Cpost,tdiag=fnp.diag(K21).astype(fnp.float32))

def feature(packet,w_next,pushed,omega,v_next):
    p=packet['p'];lam=packet['lam'];w=packet['w'];a=packet['A'];B=packet['B'];b=packet['b'];C=packet['C']
    g=packet['g'];d=packet['d'];q=int(omega.shape[1]);n=int(p.shape[0]);R=int(p.shape[1])
    assert pushed.shape[0]==n and pushed.shape[1]>=R
    z=g[:,None]*w_next
    h=d[:,None]*w_next*(C@z)
    # The two contracted CP legs reuse the actual final inherited bank.
    # The independently checked rounding difference is explicitly retained.
    pz=pushed[:,:R].T
    po=p.T@omega
    projected=po.T@(lam[:,None]*(pz*pz))
    wz=w@z;wo=w@omega
    projected=projected+((a.T@wo).T@(wz*wz)+2*wo.T@(wz*(a.T@wz)))/3
    bz=B@z
    projected=projected+(omega.T@b)[:,None]*fnp.sum(z*bz,axis=0)[None,:]
    projected=projected+2*(omega.T@bz)*(b@z)[None,:]
    rays=12*fnp.sum((omega.T@h)*projected,axis=0)/q
    # Independent fourth-moment Rademacher identity for the radial trace.
    t=g[:,None]*omega;ct=C@t;pt=p.T@t
    value=p@(lam[:,None]*pt*pt)
    wt=w@t
    value=value+w.T@(a@(wt*wt)+2*wt*(a.T@wt))/3
    bt=B@t
    value=value+b[:,None]*fnp.sum(t*bt,axis=0)[None,:]+2*bt*(b@t)[None,:]
    probe_rays=12*fnp.sum((d[:,None]*omega*ct)*value,axis=0)
    diagonal_rays=12*d*fnp.diag(C)*g*g*g*packet['tdiag']
    trace3=fnp.sum(probe_rays)/q+2*fnp.sum(diagonal_rays)
    cp=packet['Cpost']
    denominator=fnp.trace(cp)**2+2*fnp.sum(cp*cp)
    assert bool(fnp.isfinite(denominator)) and bool(denominator>0)
    radial=trace3/denominator
    delta=rays.astype(fnp.float64)-radial*v_next*v_next
    assert bool(fnp.all(fnp.isfinite(delta)))
    return delta,dict(rays=rays,radial=radial,trace3=trace3,probe_rays=probe_rays,
                      denominator=denominator,delta=delta)
