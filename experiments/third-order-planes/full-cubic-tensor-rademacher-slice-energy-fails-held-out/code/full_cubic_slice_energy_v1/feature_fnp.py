"""Fixed bilateral probes of the entire rounded auxiliary cubic tensor.

No targets, weights-specific probe selection, numerical cache, or fitted values.
The signed CP pairs are combined algebraically before cancellation can grow.
"""
import flopscope.numpy as fnp
from probes import G_SIGNS,H_SIGNS

def probes():
    # Charged anew inside each prediction, even for a warm estimator instance.
    return (fnp.asarray(G_SIGNS,dtype=fnp.float32)*0.03125,
            fnp.asarray(H_SIGNS,dtype=fnp.float32)*0.03125)

def slices(pp,lam,w,C,alive,A,b,sigma,cohorts,G,H):
    n=int(w.shape[0]);q=int(G.shape[1])
    assert G.shape==H.shape==(n,q) and pp.shape[1]==len(lam)
    pos=[];neg=[];end=0
    for start,width in cohorts:
        assert start==end and width>0 and width%2==0
        half=width//2;pos.extend(range(start,start+half));neg.extend(range(start+half,start+width));end=start+width
    assert end==len(lam)
    pos=fnp.asarray(pos,dtype=fnp.int64);neg=fnp.asarray(neg,dtype=fnp.int64)
    coeff=lam[pos]
    assert bool(fnp.all(lam[neg]==-coeff))
    sig=fnp.asarray(sigma,dtype=fnp.float32)
    assert bool(fnp.all(fnp.isfinite(sig))) and bool(fnp.all(sig>0))
    gn=G/sig[:,None];hn=H/sig[:,None]
    plus=pp[:,pos];minus=pp[:,neg]
    center=(plus+minus)*0.5;difference=(plus-minus)*0.5
    del plus,minus
    cg=center.T@gn;dg=difference.T@gn
    ch=center.T@hn;dh=difference.T@hn
    cp=2*(center@(coeff[:,None]*(cg*dh+dg*ch)))
    cp=cp+2*(difference@(coeff[:,None]*(cg*ch+dg*dh)))
    del center,difference,cg,dg,ch,dh
    mask=alive[:,None]*alive[None,:]
    a=fnp.asarray(A,dtype=fnp.float32)*mask
    fnp.fill_diagonal(a,fnp.diag(A))
    wg=w@gn;wh=w@hn
    star=w.T@(a@(wg*wh)+wg*(a.T@wh)+wh*(a.T@wg))/3
    del a
    c=fnp.asarray(C,dtype=fnp.float32)
    c=((c+c.T)*0.5)*mask
    bg=w.T@(c@wg);bh=w.T@(c@wh)
    b=fnp.asarray(b,dtype=fnp.float32)
    trace=b[:,None]*fnp.sum(gn*bh,axis=0)[None,:]
    trace=trace+bg*(b@hn)[None,:]+bh*(b@gn)[None,:]
    z=(cp+star+trace)/sig[:,None]
    energy=fnp.sum(z*z,axis=1)/q
    assert bool(fnp.all(fnp.isfinite(z))) and bool(fnp.all(fnp.isfinite(energy)))
    return z,energy
