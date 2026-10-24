"""Read saved predictions only; localize the failed finite-response gate."""
import hashlib,json,resource
from pathlib import Path
HERE=Path(__file__).resolve().parent
seal=json.loads((HERE/'diagnostic_sealed.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for r in seal['pins']:assert sha(r['path'])==r['sha256']
import numpy as np
from threadpoolctl import threadpool_info
assert all(x['num_threads']==1 for x in threadpool_info())
a={label:np.load(HERE/f'native_rank0035_{label}/prediction.npy',allow_pickle=False)
   for label in ('zero','plus','minus','halfplus','halfminus','repeat')}
assert a['halfplus'].tobytes()==a['repeat'].tobytes()
assert all(x[:2].tobytes()==a['zero'][:2].tobytes() for x in a.values())
h=1/32;rows=[]
for layer in range(16):
    v={k:x[layer].astype(np.float64) for k,x in a.items()}
    full=(v['plus']-v['minus'])/(2*h);half=(v['halfplus']-v['halfminus'])/h
    norm=float(np.linalg.norm(half));odd=float(np.linalg.norm(v['plus']-v['minus']))
    rows.append(dict(layer=layer,signal_rms=float(np.sqrt(np.mean((h*half)**2))),
        stability=float(np.linalg.norm(full-half)/norm) if norm else None,
        curvature=float(np.linalg.norm(v['plus']+v['minus']-2*v['zero'])/odd) if odd else None,
        maximum_plus_parent_change=float(np.max(np.abs(v['plus']-v['zero'])))))
q=json.loads((HERE/'qualification35.json').read_text())['response_rows'][0]
assert rows[-1]['signal_rms']==q['signal_rms'] and rows[-1]['stability']==q['stability'] and rows[-1]['curvature']==q['curvature']
result=dict(completed=True,rows=rows,first_layer_stability_above_5pct=next((r['layer'] for r in rows if r['stability'] is not None and r['stability']>.05),None),
    whole_prediction_repeat_byte_exact=True,first_two_response_rows_exact_zero=True,
    truth_arrays_read=0,new_model_calls=0,new_fit=0,internal_cause_identified=False,
    unchanged_failed_gate=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'response_diagnostic.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
