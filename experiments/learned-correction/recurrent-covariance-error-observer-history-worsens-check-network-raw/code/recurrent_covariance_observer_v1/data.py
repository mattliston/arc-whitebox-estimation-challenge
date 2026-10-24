import math,zipfile
import numpy as np
import torch
from common import *
from projection import project


UPPER=np.triu_indices(1024,1)
def pack(matrix):
    assert matrix.shape==(1024,1024) and np.all(np.diag(matrix)==0)
    assert np.allclose(matrix,matrix.T,rtol=0,atol=1e-14)
    return matrix[UPPER]
def unpack(vector):
    matrix=np.zeros((1024,1024),dtype=vector.dtype);matrix[UPPER]=vector
    return matrix+matrix.T


def layer_array(path,key,index):
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream);assert version in ((1,0),(2,0))
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=reader(stream)
        assert not fortran and not dtype.hasobject and shape[0]==16 and 0<=index<16
        size=math.prod(shape[1:])*dtype.itemsize;stream.seek(stream.tell()+index*size)
        raw=stream.read(size);assert len(raw)==size
        return np.frombuffer(raw,dtype=dtype).reshape(shape[1:]).astype(np.float64)


def packet(mini):
    wp,_=weights(mini);W=np.load(wp,mmap_mode='r');cells=[]
    with np.load(HERE/'capture'/f'mini{mini:04d}.npz',allow_pickle=False) as z:
        for l in LAYERS:
            a={k:torch.from_numpy(z[f'{k}_{l}'].copy()) for k in ('C','K','sigma','gate','pdf','F','m')}
            a['W']=torch.from_numpy(W[l].copy());cells.append(a)
        parent=torch.from_numpy(z['prediction'][-1].copy())
    with np.load(PREV/'capture'/f'mini{mini:04d}.npz') as z:
        mu=torch.from_numpy(z['mu'].copy());s=torch.sqrt(torch.from_numpy(z['v'].copy()))
    alpha=mu/s
    terminal=dict(W=torch.from_numpy(W[15].copy()),sigma=s,gate=.5*(1+torch.erf(alpha/math.sqrt(2))),
                  pdf=torch.exp(-.5*alpha**2)/math.sqrt(2*math.pi),parent=parent)
    return cells,terminal


def truth(mini,cells):
    receipt=read(OLD/f'download_mini{mini:04d}.json');entry=receipt['payload'];assert pin(entry['path'])==entry
    identity=read(OLD/f'capture_mini{mini:04d}.json')['provenance']['identity'];path=entry['path']
    with np.load(path,allow_pickle=False) as z:
        count=int(z['n_samples'].item());ordinal=int(z['global_index'].item());rawname=z['mlp_name'].item();name=rawname.decode() if isinstance(rawname,bytes) else str(rawname)
    assert count==100_000_000 and ordinal==mini and name==identity['mlp_name']
    wp,_=weights(mini);W=np.load(wp,mmap_mode='r');targets=[];rows=[]
    for l,a in zip(LAYERS,cells):
        mean=layer_array(path,'mean',l);M=layer_array(path,'M11',l);m2=layer_array(path,'post_m2',l)
        premean=layer_array(path,'pre_mean',l+1);preM=layer_array(path,'pre_M11',l+1);prem2=layer_array(path,'pre_m2',l+1)
        diagonal=max(float(np.max(np.abs(np.diag(M)-m2)/(1+np.abs(m2)))),float(np.max(np.abs(np.diag(preM)-prem2)/(1+np.abs(prem2)))))
        assert diagonal<=2e-6,(mini,l,diagonal)
        C=M-np.outer(mean,mean);preC=preM-np.outer(premean,premean);w=W[l+1].astype(float)
        emean=float(np.linalg.norm(w.T@mean-premean)/max(np.linalg.norm(premean),1e-30))
        ecov=float(np.linalg.norm(w.T@C@w-preC)/max(np.linalg.norm(preC),1e-30))
        assert emean<=1e-4 and ecov<=1e-4,(mini,l,emean,ecov)
        T=project(torch.from_numpy(C)-a['C'],a['m']);energy=float(T.square().mean())
        assert energy>0 and torch.isfinite(T).all()
        witness=float(torch.max(torch.abs(T@(a['m']/torch.linalg.vector_norm(a['m'])))))
        assert witness<1e-11 and torch.count_nonzero(torch.diagonal(T))==0
        targets.append(T)
        rows.append(dict(layer=l,moment_diagonal_scaled_max=diagonal,affine_mean_relative=emean,affine_covariance_relative=ecov,projection_annihilation_max_abs=witness,energy=energy))
    final=torch.from_numpy(layer_array(path,'gt_mean',15))
    assert final.shape==(1024,) and torch.isfinite(final).all()
    return targets,final,dict(mini=mini,reference=entry,n_samples=count,final_target='copied public1B gt_mean[15]',cells=rows)


def saved_targets(mini):
    with np.load(HERE/'targets'/f'mini{mini:04d}.npz') as z:
        return [torch.from_numpy(unpack(z[f'T_{l}_upper'])) for l in LAYERS],torch.from_numpy(z['final'].copy())
