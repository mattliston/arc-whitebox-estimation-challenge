"""Independent full-width NumPy recurrence and scalar-summation score replay."""
import math
import numpy as np
from common import *
from numpy_replay import replay
verify();verify('unit_seal.json');verify('fit_seal.json')
r=read(HERE/'result.json');frozen=read(HERE/'predictions_frozen.json');assert not frozen['check_targets_accessed']
for name in ('capture','unit','fit'):
    g=read(HERE/f'{name}_guard.json');assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
for e in list(r['models'].values())+[r['normalizers']]:assert pin(e['path'])==e
for row in r['reference_checks']:
    assert pin(row['reference']['path'])==row['reference'] and pin(row['target']['path'])==row['target']
    assert len(row['cells'])==7
    for a in row['cells']:assert a['affine_mean_relative']<=1e-4 and a['affine_covariance_relative']<=1e-4 and a['moment_diagonal_scaled_max']<=2e-6
with np.load(HERE/'normalizers.npz') as z:norm={k:z[k] for k in z.files}
models={}
for mode in ('history','reset'):
    with np.load(HERE/f'model_{mode}.npz') as z:models[mode]={k:z[k] for k in z.files}
upper=np.triu_indices(1024,1)
def unpack(v):
    D=np.zeros((1024,1024));D[upper]=v;return D+D.T
def square_sum(v):return math.fsum(float(x)*float(x) for x in v.ravel())
rows=[];scores=[];score_diff=[]
capture=read(HERE/'capture/complete.json')
for mini in ALL:
    row=next(x for x in capture['rows'] if x['mini']==mini);assert pin(row['packet']['path'])==row['packet']
    wp,_=weights(mini);W=np.load(wp,mmap_mode='r');cells=[]
    with np.load(row['packet']['path']) as z:
        prediction=z['prediction'].copy()
        for l in LAYERS:
            a={k:z[f'{k}_{l}'] for k in ('C','K','sigma','gate','pdf','F','m')};a['W']=W[l].copy();cells.append(a)
    with np.load(PREV/'baseline'/f'mini{mini:04d}.npz') as z:assert prediction.tobytes()==z['prediction'].tobytes()
    with np.load(PREV/'capture'/f'mini{mini:04d}.npz') as z:
        sigma=np.sqrt(z['v']);alpha=z['mu']/sigma
    terminal=dict(W=W[15].copy(),sigma=sigma,gate=np.array([.5*(1+math.erf(float(a)/math.sqrt(2))) for a in alpha]),pdf=np.exp(-.5*alpha**2)/math.sqrt(2*math.pi),parent=prediction[-1])
    for mode,P in models.items():
        e=next(x['packet'] for x in frozen['rows'] if x['mini']==mini and x['mode']==mode);assert pin(e['path'])==e
        Ds,correction,forecast=replay(P,cells,terminal,norm,mode)
        with np.load(e['path']) as p,np.load(HERE/'targets'/f'mini{mini:04d}.npz') as t:
            errors=[];ratios=[];constraints=[]
            for l,D,a in zip(LAYERS,Ds,cells):
                saved=unpack(p[f'D_{l}_upper']);gap=D-saved
                rel=float(np.linalg.norm(gap)/max(np.linalg.norm(saved),1e-30));absolute=float(np.max(np.abs(gap)))
                assert rel<1e-4 and absolute<1e-8,(mini,mode,l,rel,absolute)
                m=a['m']/np.linalg.norm(a['m']);constraint=float(np.max(np.abs(saved@m)));assert constraint<1e-11
                errors.append((rel,absolute));constraints.append(constraint)
                ratios.append(square_sum(p[f'D_{l}_upper']-t[f'T_{l}_upper'])/square_sum(t[f'T_{l}_upper']))
            e_relative=float(np.linalg.norm(correction-p['e'])/max(np.linalg.norm(p['e']),1e-30))
            y_abs=float(np.max(np.abs(forecast-p['forecast'])))
            assert e_relative<1e-4 and y_abs<1e-8,(mini,mode,e_relative,y_abs)
            mse=square_sum(p['forecast']-t['final'])/1024
            parent=square_sum(prediction[-1]-t['final'])/1024
            old=next(x for x in r['rows'] if x['mini']==mini and x['mode']==mode)
            score_diff.extend([abs(mse/parent-old['raw_ratio']),abs(sum(ratios)/7-old['covariance_ratio'])])
            scores.append(dict(mini=mini,mode=mode,mse=mse,parent_mse=parent,raw_ratio=mse/parent,covariance_ratio=sum(ratios)/7))
            rows.append(dict(mini=mini,mode=mode,state_max_relative=max(x[0] for x in errors),state_max_abs=max(x[1] for x in errors),mean_correction_relative=e_relative,forecast_max_abs=y_abs,constraint_max_abs=max(constraints)))
    print(json.dumps(dict(event='verified_network',mini=mini)),flush=True)
h=[x for x in scores if x['mini'] in CHECK and x['mode']=='history'];reset=[x for x in scores if x['mini'] in CHECK and x['mode']=='reset']
pooled=sum(x['mse'] for x in h)/sum(x['parent_mse'] for x in h);incremental=sum(x['mse'] for x in h)/sum(x['mse'] for x in reset)
gates=dict(pooled_raw=pooled<=.90,each_raw=all(x['raw_ratio']<=.95 for x in h),incremental=incremental<=.975,both_vs_reset=all(x['mse']<y['mse'] for x,y in zip(h,reset)),covariance=sum(x['covariance_ratio'] for x in h)/2<=.90)
assert gates==r['gates'] and all(gates.values())==r['passed'] and max(score_diff)<1e-12
log=[json.loads(line) for line in (HERE/'training.jsonl').read_text().splitlines()]
assert len(log)==60
for mode in models:
    arm=[x for x in log if x['mode']==mode];assert len(arm)==30 and [x['updates'] for x in arm]==list(range(6,181,6))
    memory=max(x['maximum_memory_gradient'] for x in arm)
    assert memory>0 if mode=='history' else memory==0
once(HERE/'verification.json',dict(passed=True,study_passed=r['passed'],gates=gates,rows=rows,maximum_score_difference=max(score_diff),result_sha256=sha(HERE/'result.json'),verifier=pin(Path(__file__))))
print(json.dumps(dict(verified=True,study_passed=r['passed'],score_max=max(score_diff),gates=gates)))
