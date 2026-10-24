"""Independent scalar reductions of the preserved consumer assessment."""
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent
assert not (H/'verification.json').exists()
a=json.loads((H/'assessment.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(H/'values.npz')==a['values_sha256']
assert sha(H/'capture.npz')==a['capture_sha256']
z=np.load(H/'values.npz',allow_pickle=False)
errors={}
for name,record in a['consumers'].items():
    ta=z[f'target_A_{name}'].ravel();tb=z[f'target_B_{name}'].ravel();p=z[f'prediction_{name}'].ravel()
    dot=lambda x,y:math.fsum(float(a)*float(b) for a,b in zip(x,y))
    signal=dot(ta,tb);residual=dot(ta-p,tb-p)
    checks={'target_cross_energy':signal,'residual_cross_energy':residual,
            'residual_cross_ratio':residual/signal,'source_energy':dot(p,p),
            'source_target_cross':dot(p,(ta+tb)/2)}
    delta=max(abs(v-record[k])/max(1e-300,abs(v)) for k,v in checks.items())
    risks=[dot(ta-p,ta-p)/dot(ta,ta),dot(tb-p,tb-p)/dot(tb,tb)]
    delta=max(delta,max(abs(x-y) for x,y in zip(risks,record['ordinary_risks'])))
    assert delta<=1e-12,(name,delta)
    assert record['passed']==(signal>0 and residual/signal<=.90 and max(risks)<1)
    errors[name]=delta
assert a['passed']==all(a['consumers'][name]['passed'] for name in ('covariance','next_mean'))
out=dict(passed=True,consumer_max_relative_discrepancies=errors,
         assessment_sha256=sha(H/'assessment.json'),utc_finished=datetime.now(timezone.utc).isoformat())
with (H/'verification.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2),flush=True)
