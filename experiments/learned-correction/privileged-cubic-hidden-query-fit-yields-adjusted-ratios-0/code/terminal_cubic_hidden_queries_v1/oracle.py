import json,resource
from common import *
def solve(np,X,error):
    scale=np.maximum(np.sqrt(np.mean(X*X,axis=0)),1e-30);Z=X/scale[None,:]
    u,s,vt=np.linalg.svd(Z,full_matrices=False);keep=s>1e-10*s[0];U=u[:,keep]
    beta=(vt[keep].T@((U.T@error)/s[keep]))/scale;correction=X@beta;projected=U@(U.T@error)
    relative=float(np.linalg.norm(correction-projected)/max(np.linalg.norm(projected),1e-30));normal=float(np.linalg.norm(U.T@(error-correction))/max(np.linalg.norm(error),1e-30));assert max(relative,normal)<=1e-8,(relative,normal)
    return correction,beta,dict(rank=int(keep.sum()),singular_values=s.tolist(),scale=scale.tolist(),coefficient_projection_relative=relative,normal_relative=normal)
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info());verify();verify('features_sealed.json')
    refs=read(HERE/'controls.json');feat=read(HERE/'features_result.json');rows=[];payload={}
    for rank in (35,100):
        row=next(r for r in feat['rows'] if r['rank']==rank)
        with np.load(row['file']['path'],allow_pickle=False) as z:B=z['affine'];Z=z['feature'];pred=z['prediction'][-1].astype(np.float64)
        tp=truth(rank)
        with np.load(tp,allow_pickle=False) as z:target=z['mc_post'][-1].astype(np.float64)
        error=target-pred;f=refs[str(rank)]['flops'];base=float(np.mean(error*error));parent=dict(raw=base,flops=f,adjusted=base*max(.1,f/2**41));arms={}
        for name,X,extra in (('affine',B,1_000_000),('extended',np.column_stack([B,Z]),row['helper_flops']+1_000_000)):
            corr,coef,check=solve(np,X,error);ideal=pred+corr;served=np.maximum(ideal,0).astype(np.float32);raw=float(np.mean((ideal-target)**2));served_raw=float(np.mean((served.astype(np.float64)-target)**2));flops=f+extra
            arms[name]=dict(ideal_raw=raw,served_raw=served_raw,charged_flops=flops,ideal_adjusted=raw*max(.1,flops/2**41),served_adjusted=served_raw*max(.1,flops/2**41),checks=check)
            payload[f'r{rank}_{name}_coefficients']=coef;payload[f'r{rank}_{name}_ideal']=ideal;payload[f'r{rank}_{name}_served']=served
        x=arms['extended'];b=arms['affine'];ratios=dict(ideal_adjusted_parent=x['ideal_adjusted']/parent['adjusted'],served_adjusted_parent=x['served_adjusted']/parent['adjusted'],ideal_raw_affine=x['ideal_raw']/b['ideal_raw'],served_raw_affine=x['served_raw']/b['served_raw'])
        passed=all(v<=.8 for v in ratios.values());rec=dict(rank=rank,parent=parent,arms=arms,ratios=ratios,passed=passed,truth=pin(tp));rows.append(rec);print(json.dumps(dict(rank=rank,ratios=ratios,passed=passed)),flush=True)
    dest=HERE/'ORACLE_DO_NOT_DEPLOY.npz';np.savez(dest,**payload)
    passed=all(r['passed'] for r in rows);write(HERE/'result.json',dict(completed=True,passed=passed,rows=rows,oracle_coefficients_not_deployable=pin(dest),fitted_deployable_models=0,new_samples=0,official_score=False,qualification='Privileged per-network terminal feature-span fits on reused1B research targets. Frozen parent trajectory; real-linear readout; actual helper bill plus1MF allowance, not a deployed candidate bill.',peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));print(json.dumps(dict(passed=passed)),flush=True)
if __name__=='__main__':main()
