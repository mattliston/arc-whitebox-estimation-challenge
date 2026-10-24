"""Independent NumPy pair replay, dense constrained projection, and scores."""
import math
import numpy as np
from common import *
verify();verify('fit_seal.json');r=read(HERE/'result.json')
for name in ('unit','baseline','capture','fit'):
    g=read(HERE/f'{name}_guard.json');assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
for e in (r['model'],r['normalizers']):assert pin(e['path'])==e
for e in r['reference_checks']:
    assert pin(e['reference']['path'])==e['reference'] and pin(e['target']['path'])==e['target']
    assert e['affine_mean_relative']<=1e-4 and e['affine_covariance_relative']<=1e-4 and e['moment_diagonal_scaled_max']<=2e-6
assert not r['final_mean_targets_accessed']
frozen=read(HERE/'predictions_frozen.json');assert not frozen['check_targets_accessed'] and not frozen['final_mean_targets_accessed']
with np.load(HERE/'model.npz') as z:P={k:z[k] for k in z.files}
with np.load(HERE/'normalizers.npz') as z:norm={k:z[k] for k in z.files}
def replay(a):
    s=a['s14'];F=np.clip(a['F']/norm['feature_scale'],-8,8)
    R=(a['C']/np.outer(s,s)).astype(np.float32)/norm['rho_scale']
    S=(a['K']/np.outer(s*s,s)).astype(np.float32)/norm['third_scale']
    left=F@P['left.weight'].T;right=F@P['right.weight'].T
    pieces=[]
    for lo in range(0,len(s),64):
        hi=min(lo+64,len(s));h=left[lo:hi,None,:]+right[None,:,:]+P['bias']
        h=h+R[lo:hi,:,None]*P['edge.weight'][:,0]
        h=h+S[lo:hi,:,None]*P['edge.weight'][:,1]
        h=h+S[:,lo:hi].T[:,:,None]*P['edge.weight'][:,2]
        pieces.append((np.tanh(h)@P['output.weight'].T+P['output.bias']).squeeze(-1))
    raw=np.concatenate(pieces);A=(raw+raw.T).astype(float)*(.001*np.outer(s,s));np.fill_diagonal(A,0)
    m=a['prediction'][14];m=m/np.linalg.norm(m)
    system=np.diag(1-2*m*m)+np.outer(m,m)
    u=np.linalg.solve(system,2*(A@m))
    D=A-(np.outer(u,m)+np.outer(m,u))/2;np.fill_diagonal(D,0)
    return D
replay_rows=[];scores=[];differences=[]
for mini in ALL:
    capture=next(x for x in read(HERE/'capture/complete.json')['rows'] if x['mini']==mini)
    baseline=next(x for x in read(HERE/'baseline/complete.json')['rows'] if x['mini']==mini)
    for e in (capture['packet'],baseline['packet']):assert pin(e['path'])==e
    with np.load(capture['packet']['path']) as z:a={k:z[k] for k in ('C','K','s14','F','W','prediction')}
    with np.load(baseline['packet']['path']) as z:assert a['prediction'].tobytes()==z['prediction'].tobytes()
    e=next(x['packet'] for x in frozen['rows'] if x['mini']==mini);assert pin(e['path'])==e
    with np.load(e['path']) as z:p={k:z[k] for k in z.files}
    with np.load(HERE/'targets'/f'mini{mini:04d}.npz') as z:t={k:z[k] for k in z.files}
    D=replay(a);rel=float(np.linalg.norm(D-p['D'])/max(np.linalg.norm(p['D']),1e-30))
    absolute=float(np.max(np.abs(D-p['D'])));assert rel<1e-4 and absolute<1e-8,(mini,rel,absolute)
    m=a['prediction'][14]/np.linalg.norm(a['prediction'][14]);constraint=float(np.max(np.abs(p['D']@m)))
    assert constraint<1e-11 and np.all(np.diag(p['D'])==0) and np.allclose(p['D'],p['D'].T,rtol=0,atol=1e-14)
    witness=[]
    for col in (0,37,999):
        w=a['W'][:,col].astype(float);value=float(np.einsum('i,ij,j->',w,p['D'],w,optimize=False))
        witness.append(abs(value-p['variance'][col]))
    assert max(witness)<1e-11
    ratio={}
    for key,pk,tk in (('covariance','D','T'),('response','response','response')):
        diff=(p[pk]-t[tk]).ravel();actual=t[tk].ravel()
        ratio[key]=math.fsum(float(x)*float(x) for x in diff)/math.fsum(float(x)*float(x) for x in actual)
    old=next(x for x in r['rows'] if x['mini']==mini);differences.extend(abs(v-old['ratios'][k]) for k,v in ratio.items())
    scores.append(dict(mini=mini,ratios=ratio));replay_rows.append(dict(mini=mini,model_relative=rel,model_max_abs=absolute,constraint_max_abs=constraint,variance_witness_max_abs=max(witness)))
gates={str(x['mini']):x['ratios']['covariance']<=.90 and x['ratios']['response']<=.85 for x in scores if x['mini'] in CHECK}
assert gates==r['gates'] and all(gates.values())==r['passed'] and max(differences)<1e-12
log=[json.loads(line) for line in (HERE/'training.jsonl').read_text().splitlines()]
assert len(log)==60 and [x['updates'] for x in log]==list(range(6,361,6))
once(HERE/'verification.json',dict(passed=True,study_passed=r['passed'],gates=gates,rows=replay_rows,maximum_score_difference=max(differences),result_sha256=sha(HERE/'result.json'),verifier=pin(Path(__file__))))
print(json.dumps(dict(verified=True,study_passed=r['passed'],score_max=max(differences),model_max_relative=max(x['model_relative'] for x in replay_rows),gates=gates)))
