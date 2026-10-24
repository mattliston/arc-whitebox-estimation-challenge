"""Independent NumPy transcription of the original physical feature/trainer math."""
import numpy as np
def reference(m,K21,aux,p,wide=False):
    a=aux['alpha'].astype(np.float64);g=aux['gate'].astype(np.float64)
    q=aux['pdf'].astype(np.float64);s=aux['sigma'].astype(np.float64)
    sk=aux['pre_k3']/s**3;ku=aux['pre_k4']/s**4;ms=m/s
    dtype=np.float64 if wide else np.float32
    F=np.column_stack([g,q,a*q,(a*a-1)*q,ms,a*g,sk*g,sk*q,sk*a*q,sk*ms,
                       ku*g,ku*q,ku*a*q,ku*ms]).astype(dtype)
    R=(aux['pre_cov']/np.outer(s,s)).astype(dtype)
    S=(K21/(s[:,None]**2*s[None,:])).astype(dtype)
    t={k:v.astype(dtype) for k,v in p.items()}
    F=F/t['feature_scale'];R=R/t['rho_scale'];S=S/t['third_scale']
    h=np.expand_dims(F@t['left'],1)+np.expand_dims(F@t['right'],0)+t['bias']
    h=h+R[:,:,None]*t['edge'][0]+S[:,:,None]*t['edge'][1]+S.T[:,:,None]*t['edge'][2]
    raw=(np.tanh(h)@t['output'])*q.astype(dtype)[:,None]*g.astype(dtype)[None,:]
    return (np.asarray(.5,dtype=dtype)*(raw+raw.T)*np.outer(s.astype(dtype),s.astype(dtype))*t['cov_factor']).astype(np.float64)
