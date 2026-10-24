"""Target-free Gaussian feature normalization and initial observer payload."""
from common import HERE,D,P2,TRAIN,CHECK,read,write,pin,sha,verify
def main():
    import numpy as np
    verify('preparation_sources_sealed.json')
    all_records=[]
    for folder,watch in [('qualification','qualification_watch.json'),('capture_all','capture_all_watch.json')]:
        g=read(HERE/watch);assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
        r=read(HERE/folder/'result.json');assert r['passed'];all_records+=r['records']
    rows={r['rank']:r for r in all_records if r['rank'] in TRAIN+CHECK}
    assert set(rows)==set(TRAIN+CHECK)
    xs=[]
    for rank in TRAIN:
        r=rows[rank];assert sha(r['features']['path'])==r['features']['sha256']
        with np.load(r['features']['path'],allow_pickle=False) as z:xs.append(z['features'].astype(np.float64))
    x=np.stack(xs);centers=np.mean(x,axis=(0,2));scales=np.sqrt(np.mean((x-centers[None,:,None,:])**2,axis=(0,2)))
    ppath=D/'lazy_frozen_trajectory_observer_v1/capture_qualification/parent_P.npz'
    with np.load(ppath,allow_pickle=False) as z:P={k:z[k].copy() for k in z.files}
    P['centers']=centers.astype(np.float32);P['scales']=np.maximum(scales,1e-6).astype(np.float32)
    P['head'][15]=0;P['head_b'][15]=0;P['gain'][15]=0;P['rms'][15]=.01
    assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in P.values())
    initial=HERE/'initial_P.npz';np.savez(initial,**P)
    lazy=read(D/'lazy_frozen_trajectory_observer_v1/inputs.json')['cold']
    inputs=[]
    for rank in TRAIN+CHECK:
        r=rows[rank];target=P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
        assert target.is_file()
        par=lazy[str(rank)];assert sha(par['prediction'])==par['prediction_sha256']
        inputs.append(dict(rank=rank,role='TRAIN' if rank in TRAIN else 'CHECK',features=r['features'],weights=r['weights'],
                           reference=pin(target),host_feature_flops=r['host_feature_flops'],lazy_reference=par))
    write(HERE/'inputs.json',dict(rows=inputs,initial_P=pin(initial),parent_P=pin(ppath),train=TRAIN,check=CHECK,truth_accessed=False,
        normalizers='TRAIN population mean/std over network and unit axes; scale floor1e-6, then FP32',final_rms=.01))
    print({'prepared_rows':len(inputs),'truth_accessed':False,'initial_P':pin(initial)},flush=True)
if __name__=='__main__':main()
