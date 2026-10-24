"""One fixed independent8M stream for the q4 residual-third oracle cell."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def finish(mean_sum,second_sum,slice_sum,next_sum,next_second,next_square,next_third,U,indices,count):
    m=mean_sum/count;second=second_sum/count;projected_mean=U.T@m
    M=slice_sum/count
    for a,u in enumerate(U.T):
        v=second@u
        M[a]-=projected_mean[a]*second+np.outer(m,v)+np.outer(v,m)-2*projected_mean[a]*np.outer(m,m)
    asymmetry=float(np.linalg.norm(M-M.transpose(0,2,1))/max(np.linalg.norm(M),1e-30))
    M=(M+M.transpose(0,2,1))/2
    y=next_sum/count;V=next_second/count;y2=next_square/count
    K=next_third/count-2*y[indices,None]*V-y[None,:]*y2[indices,None]+2*(y[indices]**2)[:,None]*y[None,:]
    unbiased=count*count/((count-1)*(count-2))
    return dict(slices=M*unbiased,next_K21=K*unbiased,post_mean_residual=m,
                post_covariance=second-np.outer(m,m),next_pre_mean_residual=y,
                raw_slice_asymmetry=asymmetry,third_unbiased_factor=unbiased)


def certificate():
    rng=np.random.default_rng(281913)
    X=rng.normal(size=(8192,9));X+=.1*X*X
    U=np.linalg.qr(rng.normal(size=(9,4)))[0];W=rng.normal(size=(9,9))/3
    shift=X.mean(0)+rng.normal(size=9)*.02
    h=X-shift;y=X@W-rng.normal(size=9)*.1;indices=np.arange(0,9,2)
    r=finish(h.sum(0),h.T@h,np.stack([h.T@(h*(h@u)[:,None]) for u in U.T]),
             y.sum(0),y[:,indices].T@y,(y*y).sum(0),(y[:,indices]**2).T@y,U,indices,len(h))
    z=X-X.mean(0);v=z@W;fac=r['third_unbiased_factor']
    target=np.stack([z.T@(z*(z@u)[:,None])/len(z) for u in U.T])*fac
    K=(v[:,indices]**2).T@v/len(v)*fac
    errors=dict(slices=float(np.max(np.abs(r['slices']-target))),plane=float(np.max(np.abs(r['next_K21']-K))))
    assert max(errors.values())<2e-13,errors
    return dict(passed=True,errors=errors)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stream',choices=('A','B'),required=True);args=ap.parse_args()
    out=HERE/f'residual_one_high_truth_r35_l12_{args.stream}_v1.npz';receipt=out.with_suffix('.json')
    assert not out.exists() and not receipt.exists()
    assert 'threadpoolctl' not in __import__('sys').modules
    torch.set_num_threads(4)
    small=certificate()
    assert torch.backends.mps.is_available()
    frozen_path=HERE/'residual_one_high_inputs_r35_l12_v1.npz'
    frozen_hash=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert frozen_hash==json.loads(frozen_path.with_suffix('.json').read_text())['output_sha256']
    f=np.load(frozen_path);U=f['U'];indices=f['indices']
    DEV=torch.device('mps');P=8000000;BATCH=65536;RANK=35;LAYER=12
    STORE=HERE.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
    entry=json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][RANK]
    rmp=STORE/entry['path'];seed=int(json.loads(rmp.read_text())['mlp_seed'])
    weights=np.load(rmp.parent/'weights.npy',mmap_mode='r')
    wg=[torch.from_numpy(np.array(weights[l],np.float32,copy=True)).to(DEV) for l in range(LAYER+2)]
    sg=torch.tensor(f['post_shift'],dtype=torch.float32,device=DEV)
    yg=torch.tensor(f['next_pre_shift'],dtype=torch.float32,device=DEV)
    ug=torch.tensor(U,dtype=torch.float32,device=DEV);actual_U=ug.cpu().numpy().astype(np.float64)
    ig=torch.tensor(indices,dtype=torch.long,device=DEV)
    direction_relative=float(np.linalg.norm(actual_U-U)/np.linalg.norm(U))
    assert direction_relative<5e-8
    mean_sum=np.zeros(1024);second_sum=np.zeros((1024,1024));slice_sum=np.zeros((4,1024,1024))
    slice_diagonal_sum=np.zeros((4,1024))
    next_sum=np.zeros(1024);next_second=np.zeros((len(indices),1024))
    next_square=np.zeros(1024);next_third=np.zeros_like(next_second)
    xor=0x4F4E454849474831 if args.stream=='A' else 0x4F4E454849474832
    rng=np.random.Generator(np.random.PCG64DXSM((seed^xor)&((1<<64)-1)))
    start=time.time();done=0
    with torch.no_grad():
        while done<P:
            count=min(BATCH,P-done)
            x=torch.from_numpy(rng.standard_normal((count,1024),dtype=np.float32)).to(DEV)
            for w in wg[:-1]:x=torch.clamp(x@w,min=0)
            h=x-sg;v=h@ug
            mean_sum+=h.sum(0).cpu().numpy().astype(np.float64)
            second_sum+=(h.T@h).cpu().numpy().astype(np.float64)
            for a in range(4):
                weighted=h*v[:,a:a+1]
                slice_sum[a]+=(h.T@weighted).cpu().numpy().astype(np.float64)
            slice_diagonal_sum+=((h*h).T@v).T.cpu().numpy().astype(np.float64)
            y=x@wg[-1]-yg;yi=y.index_select(1,ig)
            next_sum+=y.sum(0).cpu().numpy().astype(np.float64)
            next_second+=(yi.T@y).cpu().numpy().astype(np.float64)
            next_square+=(y*y).sum(0).cpu().numpy().astype(np.float64)
            next_third+=((yi*yi).T@y).cpu().numpy().astype(np.float64)
            done+=count
            if done%(BATCH*16)==0:print(json.dumps(dict(stream=args.stream,samples=done,wall_s=time.time()-start)),flush=True)
    diagonal_error=float(np.linalg.norm(np.diagonal(slice_sum,axis1=1,axis2=2)-slice_diagonal_sum)/np.linalg.norm(slice_diagonal_sum))
    assert diagonal_error<1e-4,diagonal_error
    result=finish(mean_sum,second_sum,slice_sum,next_sum,next_second,next_square,next_third,actual_U,indices,P)
    assert result['raw_slice_asymmetry']<1e-5,result['raw_slice_asymmetry']
    # MPS uses rounded basis vectors; retain and report them, do not present
    # the two-stream point estimates as removal of shared arithmetic bias.
    np.savez(out,**result,U_actual=actual_U,indices=indices,P=P,seed=seed,seed_xor=np.uint64(xor),batch=BATCH,
             post_shift_actual=sg.cpu().numpy(),next_shift_actual=yg.cpu().numpy())
    record=dict(stream=args.stream,rank=RANK,layer=LAYER,samples=P,batch=BATCH,seed_xor=hex(xor),
                small_certificate=small,frozen_input_sha256=frozen_hash,basis_rounding_relative=direction_relative,
                raw_slice_diagonal_consistency=diagonal_error,raw_slice_asymmetry=result['raw_slice_asymmetry'],
                third_unbiased_factor=result['third_unbiased_factor'],wall_s=time.time()-start,
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),torch_version=torch.__version__,
                output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                qualification='New independent seed law, disjoint from archived host/truth seeds and the other stream. Fixed q4 postL12 slices and32 next-K21 rows only. Centered MPS f32 products/f64 accumulation; shared arithmetic bias not measured. No score.')
    receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
