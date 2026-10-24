"""Independent replay of frozen fold payloads and scalar score aggregation."""
from pathlib import Path
import hashlib
import json
import math
import resource
import numpy as np

HERE=Path(__file__).resolve().parent;H2=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((HERE/'seal.json').read_text())
for name,digest in seal['sources'].items():assert sha(HERE/name)==digest
p=json.loads((HERE/'protocol.json').read_text());ranks=p['ranks']
g=json.loads((HERE/'fit_guard.json').read_text())
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
a=json.loads((HERE/'fit/result.json').read_text())
assert sha(HERE/'fit/predictions.npz')==a['predictions_sha256']
with np.load(HERE/'fit/predictions.npz') as z:
    data={k:z[k] for k in z.files}
assert data['ranks'].tolist()==ranks
capture=json.loads((HERE/'capture_complete.json').read_text());rows={r['rank']:r for r in capture['rows']}
oldpath=H2/'native_final_graph_head_v1/capture_complete.json'
old=json.loads(oldpath.read_text());oldrows={r['rank']:r for r in old['rows']}
x=[];s=[];all20parity=[];historical_bill_differences={}
for rank in ranks:
    path=HERE/rows[rank]['path'];assert sha(path)==rows[rank]['sha256']
    with np.load(path) as z:
        x.append(np.concatenate((z['control'],z['messages']),axis=1));s.append(z['sigma'])
        op=oldpath.parent/oldrows[rank]['path'];assert sha(op)==oldrows[rank]['sha256']
        with np.load(op) as oz:
            exact=bool(np.array_equal(z['prediction'],oz['prediction']) and np.array_equal(z['control'],oz['control'])
                       and np.array_equal(z['sigma'],oz['sigma']))
        assert exact,rank
        all20parity.append(rank)
        historical_bill_differences[str(rank)]=rows[rank]['flops']-oldrows[rank]['flops']
x=np.asarray(x);s=np.asarray(s)
basemse=[];mses={k:[] for k in ('control','messages')}
for i in range(20):
    basemse.append(math.fsum((float(u)-float(v))**2 for u,v in zip(data['base'][i],data['truth'][i]))/1024)
    for mode in mses:
        mses[mode].append(math.fsum((float(u)-float(v))**2 for u,v in zip(data[mode][i],data['truth'][i]))/1024)
score_error=max(abs(v-w) for v,w in zip(basemse,a['base_mse']))
for mode in mses:
    score_error=max(score_error,max(abs(v-w) for v,w in zip(mses[mode],a['scores'][mode]['network_mse'])))
    ratio=math.fsum(mses[mode])/math.fsum(basemse)
    assert abs(ratio-a['scores'][mode]['pooled_raw_ratio'])<1e-14
assert score_error<1e-20

def nn(inp,z):
    h=inp.astype(np.float64)
    for index in (0,2,4):
        h=h@z[f'layers.{index}.weight'].astype(np.float64).T+z[f'layers.{index}.bias'].astype(np.float64)
        if index<4:h=h/(1+np.exp(-h))
    return h[:,0]

max_replay=0.;covered={mode:[] for mode in mses}
for row in a['fold_models']:
    mode,fold=row['mode'],row['fold']
    it=[i for i in range(20) if i%5!=fold];ic=[i for i in range(20) if i%5==fold]
    assert row['fit_ranks']==[ranks[i] for i in it] and row['check_ranks']==[ranks[i] for i in ic]
    assert not set(row['fit_ranks'])&set(row['check_ranks'])
    mp=HERE/'fit'/f'{mode}_fold{fold}.npz';assert sha(mp)==row['weights_sha256']
    with np.load(mp) as zz:z={k:zz[k] for k in zz.files}
    train=x[it].reshape(-1,222)
    mean=train.mean(axis=0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(train.std(axis=0,dtype=np.float64),1e-6).astype(np.float32)
    assert np.array_equal(mean,z['mean']) and np.array_equal(scale,z['scale'])
    inp=np.clip((x[ic].reshape(-1,222)-mean)/scale,-8,8)
    if mode=='control':inp[:,94:]=0
    ref=np.maximum(data['base'][ic]+(nn(inp,z)*s[ic].reshape(-1)*.001).reshape(4,1024),0)
    gap=float(np.max(np.abs(ref-data[mode][ic])))
    # Independent FP64 algebra versus the frozen FP32 Torch path plus served rounding.
    tolerance=float(2*np.max(np.spacing(np.abs(data[mode][ic]).astype(np.float32))))+1e-10
    assert gap<=tolerance,(mode,fold,gap,tolerance)
    max_replay=max(max_replay,gap);covered[mode]+=ic
for indices in covered.values():assert sorted(indices)==list(range(20))
m,c=a['scores']['messages'],a['scores']['control']
gates=dict(pooled_raw=m['pooled_raw_ratio']<=.98,matched_control=m['pooled_raw_ratio']/c['pooled_raw_ratio']<=.985,
           folds=m['winning_folds']>=4,networks=m['winning_networks']>=12,worst=m['worst_network_raw_ratio']<=1.2)
assert gates==a['gates'] and a['passed']==all(gates.values())
result=dict(passed=True,independent_score_max_abs_error=score_error,independent_float64_prediction_max_abs_gap=max_replay,
            network_folds_disjoint_and_complete=True,normalizers_training_only_exact=True,
            all20_parent_predictions_controls_and_sigmas_exact=all20parity,
            process_matched_8_parent_bills_exact=capture['independent_controls_exact'],
            historical_different_process_order_bill_differences=historical_bill_differences,
            previous_capture_manifest_sha256=sha(oldpath),fit_result_sha256=sha(HERE/'fit/result.json'),
            verification_source_sha256=sha(Path(__file__)),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'verification_v2.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
