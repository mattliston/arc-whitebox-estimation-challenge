from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
P2=HERE.parent
manifest=json.loads((HERE/'trace_vector_depth_tilt_manifest_v1.json').read_text())
base='both_tail_train2_v6'
tags=['trace_vector_depth_tilt_plus_train2_v1','trace_vector_depth_tilt_minus_train2_v1']
for arm,tag in zip(manifest['arms'],tags):
    assert hashlib.sha256(Path(arm['path']).read_bytes()).hexdigest()==arm['sha256']
    m=json.loads((HERE/tag/'manifest.json').read_text())
    assert m['sha256']==arm['sha256'] and m['ranks']==[35,100]
    g=json.loads((HERE/(tag+'_guard.json')).read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
rows=[];errors=[];directions=[]
for rank in (35,100):
    b=np.load(HERE/base/f'pred_{rank}.npy')
    p=np.load(HERE/tags[0]/f'pred_{rank}.npy')
    m=np.load(HERE/tags[1]/f'pred_{rank}.npy')
    truth=np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
    assert np.array_equal(b[0],p[0]) and np.array_equal(b[0],m[0])
    e=b[15]-truth; odd=(p[15]-m[15])/2; d=odd/.02
    rem=(p[15]+m[15])/2-b[15]
    ratio=float(np.linalg.norm(rem)/np.linalg.norm(odd))
    rows.append(dict(rank=rank,even_to_odd_ratio=ratio,
        odd_rms=float(np.sqrt(np.mean(odd**2))),even_rms=float(np.sqrt(np.mean(rem**2))),
        plus_raw_pct=100*(np.mean((p[15]-truth)**2)/np.mean(e**2)-1),
        minus_raw_pct=100*(np.mean((m[15]-truth)**2)/np.mean(e**2)-1),
        first_layer_byte_exact=True))
    errors.append(e);directions.append(d)
e=np.concatenate(errors);d=np.concatenate(directions)
theta=float(np.clip(-np.dot(e,d)/np.dot(d,d),-.04,.04))
gain=float(100*(1-np.mean((e+theta*d)**2)/np.mean(e**2)))
passed=all(r['even_to_odd_ratio']<=.2 for r in rows) and gain>=2
result=dict(scope='Reused TRAIN2 finite-direction information only; not an evaluated estimator at theta or a general upper bound',
    rows=rows,unclipped_theta=float(-np.dot(e,d)/np.dot(d,d)),clipped_theta=theta,
    linearized_raw_gain_pct=gain,information_gate_passed=passed,
    decision='Admit one frozen-theta native TRAIN2 check' if passed else 'Close this fixed depth contrast; no finer step/profile/layer fit or validation')
with (HERE/'trace_vector_depth_tilt_information_v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result,indent=2))
