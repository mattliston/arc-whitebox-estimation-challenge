import numpy as np
def covariance(H,R,s,q,wide=False):
    dt=np.float64 if wide else np.float32
    f=np.concatenate((np.asarray(H,dtype=dt),np.ones((len(H),1),dtype=dt)),axis=1)
    f=(f/np.asarray(q['feature_scale'],dtype=dt))*(np.asarray(s,dtype=dt)/dt(q['sigma_scale']))[:,None]
    r=np.asarray(R,dtype=dt);co=np.asarray(q['theta'],dtype=dt)
    if wide:
        # Explicit feature contractions, independently of the native two GEMMs.
        terms=[np.einsum('ia,ab,jb->ij',f,co[k],f,optimize=True) for k in range(3)]
    else:terms=[(f@co[k])@f.T for k in range(3)]
    return (terms[0]+terms[1]*r+terms[2]*(r*r))*dt(q['cov_scale'])
