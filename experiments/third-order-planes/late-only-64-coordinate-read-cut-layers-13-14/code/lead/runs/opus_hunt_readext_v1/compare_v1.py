"""[opus-hunt] compare one pilot output against cached references on the same seed (net = MSE log + bill log, util floor 0.1)."""
import json, math, sys
import numpy as np
from pathlib import Path
def load(p):
    p=Path(p); r=json.loads(p.read_text())['rows'][0]
    with np.load(p.with_name(p.stem+'_pred.npz'),allow_pickle=False) as z: y=z[str(r['seed'])].copy()
    return r,y
cand=sys.argv[1]; refs=sys.argv[2:]
a,ay=load(cand)
for ref in refs:
    b,by=load(ref)
    assert a['seed']==b['seed']
    first=next((i for i in range(16) if ay[i].tobytes()!=by[i].tobytes()),16)
    mse=100*math.log(a['mse_final']/b['mse_final'])
    bill=100*math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))
    print(f"vs {Path(ref).parent.name}/{Path(ref).name}: first_diff_layer={first} dflops={a['flops']-b['flops']:+d} bill={bill:+.4f}% MSE={mse:+.4f}% NET={mse+bill:+.4f}%")
    print('  per-layer MSE log %:',' '.join(f'{100*math.log(x/y):+.3f}' for x,y in zip(a['mse_layers'],b['mse_layers'])))
st=a.get('stats',{})
for k in ('_ASTRA_READ64_STATS','_HUNT_EARLY_STATS'):
    if k in st: print(k, json.dumps(st[k])[:600])
