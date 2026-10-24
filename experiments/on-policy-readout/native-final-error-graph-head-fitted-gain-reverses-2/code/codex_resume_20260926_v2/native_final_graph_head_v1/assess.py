"""Open CHECK32 labels once, after both terminal models/predictions are sealed."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent;P2=HERE.parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((HERE/'protocol.json').read_text())
seal=json.loads((HERE/'pre_fit_source_seal.json').read_text())
for name,value in seal['files'].items():assert sha(HERE/name)==value,(name,'changed')
capture=json.loads((HERE/'capture_complete.json').read_text())
rows={r['rank']:r for r in capture['rows']}
ranks=protocol['check_ranks'];base=[];truth=[]
models={};pred={}
for mode in ('graph','control'):
    g=json.loads((HERE/f'fit_{mode}_guard.json').read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    path=HERE/f'fit_{mode}';r=json.loads((path/'result.json').read_text())
    assert not r['check_labels_accessed'] and not r['valid_labels_accessed']
    assert r['terminal_epoch']==protocol['training']['epochs']
    assert r['weights_sha256']==sha(path/'weights.npz')
    assert r['predictions_sha256']==sha(path/'check_predictions.npy')
    assert r['fit_script_sha256']==seal['files']['fit.py'] and r['model_sha256']==seal['files']['model.py']
    models[mode]=r;pred[mode]=np.load(path/'check_predictions.npy')
    assert pred[mode].shape==(len(ranks),1024)
# Freeze this audit before any CHECK labels are opened.
precheck=dict(models=models,protocol_sha256=sha(HERE/'protocol.json'),assessment_sha256=sha(HERE/'assess.py'),check_labels_accessed=False)
with (HERE/'precheck_seal.json').open('x') as f:json.dump(precheck,f,indent=2,allow_nan=False)
for rank in ranks:
    packet=HERE/rows[rank]['path'];assert sha(packet)==rows[rank]['sha256']
    with np.load(packet) as z:base.append(z['prediction'][15])
    assert rank not in protocol['fit_ranks'] and rank not in protocol['valid_ranks_not_accessed']
    with np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz') as z:truth.append(z['mc_post'][15].astype(np.float64))
base=np.stack(base);truth=np.stack(truth)
mse={'bank':np.mean((base-truth)**2,axis=1)}
for mode in pred:mse[mode]=np.mean((pred[mode]-truth)**2,axis=1)
rng=np.random.default_rng(2026092707);indices=rng.integers(0,len(ranks),size=(50000,len(ranks)))
def compare(a,b):
    ratio=float(a.mean()/b.mean())
    ci=np.quantile(a[indices].mean(1)/b[indices].mean(1),[.025,.975]).tolist()
    return dict(raw_ratio=ratio,raw_pct=100*(ratio-1),ci95_ratio=ci,ci95_pct=[100*(c-1) for c in ci],wins=int(np.sum(a<b)))
message=compare(mse['graph'],mse['bank']);control=compare(mse['control'],mse['bank']);contrast=compare(mse['graph'],mse['control'])
cfg=protocol['gates']
gates=dict(raw=bool(message['raw_ratio']<=cfg['raw_ratio_max']),
           confidence=bool(message['ci95_ratio'][1]<cfg['ci95_upper_ratio_max']),
           wins=bool(message['wins']>=cfg['wins_min']),
           source_vs_control=bool(contrast['raw_ratio']<=cfg['vs_local_control_ratio_max']))
result=dict(protocol_sha256=sha(HERE/'protocol.json'),assessment_sha256=sha(HERE/'assess.py'),
            precheck_seal_sha256=sha(HERE/'precheck_seal.json'),check_ranks=ranks,
            graph_vs_bank=message,control_vs_bank=control,graph_vs_control=contrast,
            gates=gates,passes_information_gate=all(gates.values()),
            decision='Advance frozen graph model to metered native qualification' if all(gates.values()) else
                'Close fixed final-error graph pilot; no sign, epoch, amplitude, architecture or feature rescue on CHECK32',
            per_network=[dict(rank=r,**{k:float(v[i]) for k,v in mse.items()}) for i,r in enumerate(ranks)],
            check_labels_accessed=True,valid40_accessed=False,
            native_feature_cost_or_output_parity_qualified=False,submission_qualified=False)
serialized=json.dumps(result,indent=2,allow_nan=False)
with (HERE/'assessment.json').open('x') as f:f.write(serialized+'\n')
print(serialized,flush=True)
