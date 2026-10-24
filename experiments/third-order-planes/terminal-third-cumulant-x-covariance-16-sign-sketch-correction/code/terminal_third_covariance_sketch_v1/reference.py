"""Research FP64 replay of the declared finite sketch, without native CP reuse."""
def reference(packet,w_next,omega,v_next):
    import numpy as np
    x={k:np.asarray(v,dtype=np.float64) for k,v in packet.items()}
    p=x['p'];lam=x['lam'];w=x['w'];a=x['A'];B=x['B'];b=x['b'];C=x['C'];g=x['g'];d=x['d']
    om=np.asarray(omega,dtype=np.float64);wn=np.asarray(w_next,dtype=np.float64);q=om.shape[1]
    def action(z):
        Pz=p.T@z;wz=w@z;bz=B@z
        ans=p@(lam[:,None]*Pz*Pz)
        ans+=w.T@(a@(wz*wz)+2*wz*(a.T@wz))/3
        ans+=b[:,None]*np.sum(z*bz,axis=0)[None,:]+2*bz*(b@z)[None,:]
        return ans
    z=g[:,None]*wn;hh=d[:,None]*wn*(C@z)
    projected=np.concatenate([om.T@action(z[:,i:i+32]) for i in range(0,z.shape[1],32)],axis=1)
    rays=12*np.sum((om.T@hh)*projected,axis=0)/q
    t=g[:,None]*om;pr=12*np.sum(d[:,None]*om*(C@t)*action(t),axis=0)
    # Recover the independent complete tensor diagonal from its owners.
    diag=np.sum((w.T@a)*w.T*w.T,axis=1)+3*b*np.diag(B)
    for i in range(0,p.shape[1],512):
        chunk=p[:,i:i+512];diag+=np.sum(chunk*chunk*chunk*lam[None,i:i+512],axis=1)
    td_relative=float(np.linalg.norm(diag-x['tdiag'])/max(np.linalg.norm(diag),1e-30))
    trace3=np.mean(pr)+2*np.sum(12*d*np.diag(C)*g**3*diag)
    cp=x['Cpost'];den=np.trace(cp)**2+2*np.sum(cp*cp)
    radial=trace3/den
    delta=rays-radial*np.asarray(v_next,dtype=np.float64)**2
    return delta,dict(rays=rays,radial=radial,trace3=trace3,tdiag_relative=td_relative)
