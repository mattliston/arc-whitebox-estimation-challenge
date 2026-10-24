"""Privileged per-network span test; coefficients never deployment data."""
from common import HERE,sha,read,write,seal_check,guarded
seal_check();guarded('features')
meta=read(HERE/'inputs.json');manifest=read(HERE/'features_sealed.json')
assert manifest['completed'] and manifest['repeat35_exact']
for r in manifest['rows']:assert sha(r['features'])==r['features_sha256']
write(HERE/'pretarget_features_seal.json',dict(manifest_sha256=sha(HERE/'features_sealed.json'),
    original_source_seal_sha256=sha(HERE/'sources_sealed.json'),means_not_yet_decoded=True))
import numpy as np
def project(X,y):
    u,s,vh=np.linalg.svd(X,full_matrices=False)
    keep=s>1e-10*s[0];U=u[:,keep]
    beta=(vh[keep].T/s[keep])@(U.T@y)
    fitted=U@(U.T@y);independent=X@beta
    reconstruction=float(np.linalg.norm(independent-fitted)/max(np.linalg.norm(fitted),1e-30))
    normal=float(np.linalg.norm(U.T@(y-fitted))/max(np.linalg.norm(y),1e-30))
    assert reconstruction<=1e-8 and normal<=1e-8
    return fitted,dict(rank=int(keep.sum()),columns=X.shape[1],svd_projection_relative_error=reconstruction,
                       normal_relative_error=normal,singular_values=s.tolist())
rows=[]
for r in manifest['rows'][:2]:
    rank=r['rank'];tr=meta['truths'][str(rank)];assert sha(tr['path'])==tr['sha256']
    with np.load(r['features']) as z:J,B,pred=z['J'],z['B'],z['prediction']
    with np.load(tr['path']) as z:truth=z['mc_post'][-1].astype(np.float64)
    error=truth-pred[-1].astype(np.float64);raw=float(np.mean(error*error))
    assert abs(raw/meta['expected_parent_raw'][str(rank)]-1)<=1e-12
    aff,ap=project(B,error);ext,ep=project(np.column_stack([B,J]),error)
    amse=float(np.mean((error-aff)**2));emse=float(np.mean((error-ext)**2))
    rows.append(dict(rank=rank,parent_raw=raw,affine_oracle_raw=amse,extended_oracle_raw=emse,
                     extended_to_parent=emse/raw,extended_to_affine=emse/amse,
                     affine_to_parent=amse/raw,affine_projection=ap,extended_projection=ep))
result=dict(rows=rows,information_gate_passed=all(r['extended_to_parent']<=.70 and r['extended_to_affine']<=.80 for r in rows),
    same_network_privileged_oracle=True,cross_network_generalization=False,deployable_coefficients=False,
    actual_candidate_score=False,provisional_future_cost_allowance=15_000_000_000,
    source_seal_sha256=sha(HERE/'sources_sealed.json'),feature_seal_sha256=sha(HERE/'features_sealed.json'))
write(HERE/'result.json',result)
print(__import__('json').dumps([{k:v for k,v in r.items() if 'projection' not in k} for r in rows]+[dict(gate_passed=result['information_gate_passed'])]))

