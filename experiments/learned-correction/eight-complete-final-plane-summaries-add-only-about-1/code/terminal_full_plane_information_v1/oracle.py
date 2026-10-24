import resource
import numpy as np
from common import *
def solve(matrix,y):
    rms=np.sqrt(np.mean(matrix*matrix,axis=0));assert (rms>0).all()
    A=matrix/rms[None,:];U,s,Vt=np.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10
    scaled=Vt[keep].T@((U[:,keep].T@y)/s[keep]);coef=scaled/rms
    correction=matrix@coef;residual=y-correction
    normal=float(np.linalg.norm(A.T@residual)/(np.linalg.norm(A)*np.linalg.norm(y)))
    assert normal<=1e-8,normal
    return coef,dict(rank=int(keep.sum()),columns=matrix.shape[1],singular_values=s.tolist(),rms=rms.tolist(),normal_residual=normal)
def loss(pred,truth):return float(np.mean((pred-truth)**2))
def main():
    seal_check();done('features');seal_check('features_sealed.json')
    inputs=read(HERE/'inputs.json')['rows'];features=read(HERE/'features_result.json')['rows'][:2]
    records=[];payload={}
    for meta,record in zip(inputs,features):
        assert meta['rank']==record['rank'];rank=meta['rank'];check(record['file']);check(meta['truth']);check(meta['parent_prediction'])
        with np.load(record['file']['path'],allow_pickle=False) as z:X=z['X'];Z=z['Z'];prediction=z['prediction']
        assert prediction.tobytes()==np.load(meta['parent_prediction']['path'],allow_pickle=False).tobytes()
        with np.load(meta['truth']['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        base=prediction[-1].astype(np.float64);y=truth-base;raw_parent=loss(base,truth);arms={}
        for name,matrix in [('affine',X),('extended',np.column_stack([X,Z]))]:
            coef,certificate=solve(matrix,y);ideal=base+matrix@coef;served=np.maximum(ideal,0).astype(np.float32)
            arms[name]=dict(ideal_raw=loss(ideal,truth),served_raw=loss(served.astype(float),truth),**certificate)
            payload[f'{rank}_{name}_coefficients']=coef;payload[f'{rank}_{name}_ideal']=ideal;payload[f'{rank}_{name}_served']=served
        ratios={}
        for kind in ['ideal','served']:
            ratios[kind+'_parent']=arms['extended'][kind+'_raw']/raw_parent
            ratios[kind+'_affine']=arms['extended'][kind+'_raw']/arms['affine'][kind+'_raw']
        passed=all(ratios[k+'_parent']<=.85 and ratios[k+'_affine']<=.90 for k in ['ideal','served'])
        records.append(dict(rank=rank,parent_raw=raw_parent,parent_cold_flops=meta['parent_cold_flops'],
            multiplier=max(.1,meta['parent_cold_flops']/2**41),arms=arms,ratios=ratios,passed=passed))
        print(dict(rank=rank,ratios=ratios,passed=passed),flush=True)
    np.savez(HERE/'ORACLE_DO_NOT_DEPLOY.npz',**payload)
    write(HERE/'result.json',dict(passed=all(r['passed'] for r in records),rows=records,extra_inference_cost_assumed=0,
        oracle_not_deployable=True,held_out=False,official_score=False,new_native_calls=0,
        fit_scope='two separate same-network residual projections in65/73 columns; no shared fitted candidate',
        payload=pin(HERE/'ORACLE_DO_NOT_DEPLOY.npz'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':main()
