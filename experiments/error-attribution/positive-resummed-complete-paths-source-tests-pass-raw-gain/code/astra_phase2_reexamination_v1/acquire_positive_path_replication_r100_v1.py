"""One independent8M centered repeated-third-plane replication stream."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def finish(mean,second,square,third,indices,count):
    y=mean/count;V=second/count;y2=square/count
    K=third/count-2*y[indices,None]*V-y[None,:]*y2[indices,None]+2*(y[indices]**2)[:,None]*y[None,:]
    return K*(count*count/((count-1)*(count-2)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stream',required=True,choices=('A','B'));a=ap.parse_args()
    OUT=HERE/f'positive_path_replication_truth_r100_{a.stream}_v1.npz';receipt=OUT.with_suffix('.json')
    assert not OUT.exists() and not receipt.exists()
    assert 'threadpoolctl' not in __import__('sys').modules
    rng=np.random.default_rng(26951940);X=rng.normal(size=(8192,9));X+=.12*X*X
    idx=np.arange(0,9,2);Y=X-rng.normal(size=9)*.1
    result=finish(Y.sum(0),Y[:,idx].T@Y,(Y*Y).sum(0),(Y[:,idx]**2).T@Y,idx,len(Y))
    Z=X-X.mean(0);expected=(Z[:,idx]**2).T@Z/len(Z)*(len(Z)**2/((len(Z)-1)*(len(Z)-2)))
    cert=float(np.max(np.abs(result-expected)));assert cert<2e-13
    torch.set_num_threads(4);assert torch.backends.mps.is_available();dev=torch.device('mps')
    frozen=HERE/'positive_path_replication_inputs_r100_v1.npz';digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert digest==json.loads(frozen.with_suffix('.json').read_text())['output_sha256']
    f=np.load(frozen);indices=f['indices'];RANK=100;P=8000000;BATCH=65536
    store=HERE.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
    entry=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][RANK]
    rmp=store/entry['path'];seed=int(json.loads(rmp.read_text())['mlp_seed'])
    weights=np.load(rmp.parent/'weights.npy',mmap_mode='r')
    wg=[torch.from_numpy(np.array(weights[l],np.float32,copy=True)).to(dev) for l in range(14)]
    yg=torch.tensor(f['next_pre_shift'],dtype=torch.float32,device=dev);ig=torch.tensor(indices,dtype=torch.long,device=dev)
    mean=np.zeros(1024);second=np.zeros((len(indices),1024));square=np.zeros(1024);third=np.zeros_like(second);diag=np.zeros(len(indices))
    xor=0x4F4E454849474831 if a.stream=='A' else 0x4F4E454849474832
    rng=np.random.Generator(np.random.PCG64DXSM((seed^xor)&((1<<64)-1)))
    start=time.time();done=0
    with torch.no_grad():
        while done<P:
            count=min(BATCH,P-done);x=torch.from_numpy(rng.standard_normal((count,1024),dtype=np.float32)).to(dev)
            for w in wg[:-1]:x=torch.clamp(x@w,min=0)
            y=x@wg[-1]-yg;yi=y.index_select(1,ig);yi2=yi*yi
            mean+=y.sum(0).cpu().numpy().astype(np.float64)
            second+=(yi.T@y).cpu().numpy().astype(np.float64)
            square+=(y*y).sum(0).cpu().numpy().astype(np.float64)
            third+=(yi2.T@y).cpu().numpy().astype(np.float64)
            diag+=(yi2*yi).sum(0).cpu().numpy().astype(np.float64)
            done+=count
            if done%(16*BATCH)==0:print(json.dumps(dict(stream=a.stream,samples=done,wall_s=time.time()-start)),flush=True)
    diag_error=float(np.linalg.norm(third[np.arange(len(indices)),indices]-diag)/np.linalg.norm(diag));assert diag_error<1e-4
    K=finish(mean,second,square,third,indices,P)
    np.savez(OUT,next_K21=K,indices=indices,P=P,batch=BATCH,seed=seed,seed_xor=np.uint64(xor),shift_actual=yg.cpu().numpy())
    record=dict(rank=RANK,stream=a.stream,P=P,batch=BATCH,seed_xor=hex(xor),certificate_max_error=cert,
                raw_diagonal_consistency=diag_error,frozen_input_sha256=digest,wall_s=time.time()-start,
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                qualification='New disjoint stream, fixed32 rows pre13. Centered f32 products/f64 accumulation, global unbiased3; no shared arithmetic-bias bound or score.')
    receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
