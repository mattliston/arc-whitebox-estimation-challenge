"""Independent reconstruction of the one declared solution; no new fit selection."""
import json,resource
from common import *
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info());verify();verify('captures_sealed.json');verify('statistics_sealed.json');verify('audit_sources_sealed.json')
    result=read(HERE/'result.json');ps=read(HERE/'payload_sealed.json');assert sha(ps['payload']['path'])==ps['payload']['sha256']
    with np.load(ps['payload']['path'],allow_pickle=False) as z:payload={k:z[k] for k in z.files}
    data={}
    for rank in TRAIN+VALID:
        with np.load(HERE/f'statistics_{rank:04d}.npz',allow_pickle=False) as z:data[rank]={k:z[k] for k in z.files}
    nc=sum(float(data[r]['cov_energy'][j]) for r in TRAIN for j in range(7))/42
    nv=sum(float(data[r]['value_energy'][j]) for r in TRAIN for j in range(7))/42
    G=np.zeros((16,16));b=np.zeros(16)
    for r in TRAIN:
        for j in range(7):
            G+=(data[r]['cov_gram'][j]/nc+data[r]['value_gram'][j]/nv)/42
            b+=(data[r]['cov_rhs'][j]/nc+data[r]['value_rhs'][j]/nv)/42
    scale=np.maximum(np.sqrt(np.maximum(np.diag(G),0)),1e-20);M=G/np.outer(scale,scale)+.001*np.eye(16)
    L=np.linalg.cholesky(M);beta=np.linalg.solve(L.T,np.linalg.solve(L,b/scale));coef=beta/scale
    relative=float(np.linalg.norm(coef-payload['delta64'])/max(np.linalg.norm(payload['delta64']),1e-30));assert relative<=1e-9,relative
    assert np.array_equal(coef.astype(np.float32),payload['delta32'])
    q=payload['delta32'].astype(np.float64);largest=0.;checks=[]
    for r in TRAIN+VALID:
        row=next(x for x in result['rows'] if x['rank']==r);base={'cov':0.,'value':0.};loss={'cov':0.,'value':0.}
        for j in range(7):
            for k in ('cov','value'):
                d=data[r];energy=float(d[k+'_energy'][j]);linear=sum(float(q[a])*float(d[k+'_rhs'][j,a]) for a in range(16));quadratic=sum(float(q[a])*float(d[k+'_gram'][j,a,b])*float(q[b]) for a in range(16) for b in range(16));err=energy-2*linear+quadratic
                expected=row['layers'][j][k]['residual'];rel=abs(err-expected)/max(abs(expected),1e-30);largest=max(largest,rel);assert rel<=1e-9,(r,j,k,rel)
                base[k]+=energy;loss[k]+=err
        ratios={k:loss[k]/base[k] for k in base};passed=ratios['cov']<=.98 and ratios['value']<=.95
        assert passed==row['passed'];checks.append(dict(rank=r,pooled=ratios,passed=passed))
    assert result['passed']==all(x['passed'] for x in checks if x['rank'] in VALID)
    write(HERE/'audit_result.json',dict(passed=True,one_solution_reconstructed=True,coefficient_relative=relative,rounded_payload_exact=True,maximum_loss_reconstruction_relative=largest,all_gates_reproduced=True,rows=checks,additional_candidate_or_hyperparameter=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(passed=True,coefficient_relative=relative,maximum_loss_relative=largest)),flush=True)
if __name__=='__main__':main()
