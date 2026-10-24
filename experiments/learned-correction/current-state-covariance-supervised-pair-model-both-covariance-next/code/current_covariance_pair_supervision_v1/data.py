import math,zipfile
import numpy as np
import torch
from common import *
from projection import project

def packet(mini):
    with np.load(HERE/'capture'/f'mini{mini:04d}.npz',allow_pickle=False) as z:
        return {k:torch.from_numpy(z[k].copy()) for k in z.files}

def response(a,D):
    W=a['W'].double();v=(W*(D@W)).sum(0)
    s=torch.sqrt(a['v']);pdf=torch.exp(-.5*(a['mu']/s)**2)/math.sqrt(2*math.pi)
    return v,pdf*v/(2*s)

def layer_array(path,key,index):
    """Decode only the declared layer, including for the post-mean archive."""
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        assert version in ((1,0),(2,0))
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=reader(stream)
        assert not fortran and not dtype.hasobject and shape[0]==16 and 0<=index<16
        size=math.prod(shape[1:])*dtype.itemsize
        stream.seek(stream.tell()+index*size)
        raw=stream.read(size);assert len(raw)==size
        return np.frombuffer(raw,dtype=dtype).reshape(shape[1:]).astype(np.float64)

def truth_covariance(mini,a):
    receipt=read(OLD/f'download_mini{mini:04d}.json');e=receipt['payload'];assert pin(e['path'])==e
    identity=read(OLD/f'capture_mini{mini:04d}.json')['provenance']['identity']
    with np.load(e['path'],allow_pickle=False) as z:
        count=int(z['n_samples'].item());ordinal=int(z['global_index'].item());rawname=z['mlp_name'].item();name=rawname.decode() if isinstance(rawname,bytes) else str(rawname)
        assert count==100_000_000 and ordinal==mini and name==identity['mlp_name'],(count,ordinal,name,mini,identity['mlp_name'])
    mean=layer_array(e['path'],'mean',14);premean=layer_array(e['path'],'pre_mean',15)
    M=layer_array(e['path'],'M11',14);preM=layer_array(e['path'],'pre_M11',15)
    m2=layer_array(e['path'],'post_m2',14);prem2=layer_array(e['path'],'pre_m2',15)
    diagonal=max(float(np.max(np.abs(np.diag(M)-m2)/(1+np.abs(m2)))),float(np.max(np.abs(np.diag(preM)-prem2)/(1+np.abs(prem2)))))
    assert diagonal<=2e-6,(mini,diagonal)
    C=M-np.outer(mean,mean);preC=preM-np.outer(premean,premean);W=a['W'].numpy().astype(np.float64)
    emean=float(np.linalg.norm(W.T@mean-premean)/max(np.linalg.norm(premean),1e-30))
    ecov=float(np.linalg.norm(W.T@C@W-preC)/max(np.linalg.norm(preC),1e-30))
    assert emean<=1e-4 and ecov<=1e-4,(mini,emean,ecov)
    T=project(torch.from_numpy(C)-a['C'],a['prediction'][14]);var,vr=response(a,T)
    baseline_cov=float(T.square().mean());baseline_response=float(vr.square().mean())
    assert baseline_cov>0 and baseline_response>0 and torch.isfinite(T).all()
    m=a['prediction'][14];witness=float(torch.max(torch.abs(T@(m/torch.linalg.vector_norm(m)))))
    assert witness<1e-11 and bool(torch.all(torch.diagonal(T)==0))
    info=dict(mini=mini,reference=e,n_samples=count,moment_diagonal_scaled_max=diagonal,affine_mean_relative=emean,affine_covariance_relative=ecov,projection_annihilation_max_abs=witness,baseline_covariance_energy=baseline_cov,baseline_response_energy=baseline_response,final_mean_targets_used=False)
    return T,var,vr,info
