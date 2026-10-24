"""The observer's fixed FP64 linear definition; not a native-state Jacobian."""
import numpy as np

def raw_features(a):return np.concatenate([a['features'][8:],a['hidden'][8:]],axis=2).astype(np.float64)

def local_design(raw,sigma):
    layers,n,d=raw.shape;b=np.zeros((layers,n,d+layers),np.float64)
    b[:,:,:d]=sigma[:,:,None]*raw
    for l in range(layers):b[l,:,d+l]=sigma[l]
    return b

def raw_design(raw,sigma,gate,weights):
    local=local_design(raw,sigma);b=local.copy()
    for l in range(1,len(raw)):b[l]=gate[l,:,None]*(weights[l].T@b[l-1])+local[l]
    return b

def normalize_design(b,mean,scale):
    d=len(mean);out=b.copy();constant=np.sum(b[:,:,d:],axis=2)
    out[:,:,:d]=(b[:,:,:d]-constant[:,:,None]*mean)/scale
    return out

def direct(raw,sigma,gate,weights,mean,scale,beta,transport=True):
    d=len(mean);rows=[];previous=None
    for l in range(len(raw)):
        source=((raw[l]-mean)/scale)@beta[:d]+beta[d+l]
        current=sigma[l]*source
        if transport and previous is not None:current=gate[l]*(weights[l].T@previous)+current
        rows.append(current);previous=current
    return np.stack(rows)

def normalizers(arrays):
    count=sum(a.size//a.shape[-1] for a in arrays)
    mean=sum(np.sum(a,axis=(0,1)) for a in arrays)/count
    var=sum(np.sum((a-mean)**2,axis=(0,1)) for a in arrays)/count
    return mean,np.maximum(np.sqrt(var),1e-6)

def solve_gram(gram,rhs,ridge=1e-3):
    scale=np.maximum(np.sqrt(np.maximum(np.diag(gram),0)),1e-12)
    g=gram/(scale[:,None]*scale[None,:])+ridge*np.eye(len(scale));r=rhs/scale
    theta=np.linalg.solve(g,r);beta=theta/scale
    relative=float(np.linalg.norm(g@theta-r)/max(np.linalg.norm(r),1e-30));assert relative<=1e-10
    return beta,dict(normal_equation_relative_error=relative,condition=float(np.linalg.cond(g)),design_rms=scale.tolist())

def fnp_step(fnp,raw,sigma,gate,weight,previous,mean,scale,beta,offset):
    d=int(mean.shape[0]);x=fnp.asarray(raw,dtype=fnp.float64)
    source=((x-mean)/scale)@beta[:d]+beta[d+offset]
    current=sigma*source
    if previous is not None:current=gate*(weight.T@previous)+current
    return current
