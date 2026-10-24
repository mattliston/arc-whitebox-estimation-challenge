"""One fixed finite endpoint; no other probe score or amplitude selection."""
import hashlib,json,resource
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
seal=read(HERE/'sources_sealed.json')
for row in seal['pins']:assert sha(row['path'])==row['sha256'],row['path']
inputs=read(HERE/'inputs.json')
assert inputs['candidate']['sha256']=='f6cbcccd7c5933aa75c71ab8e0ddf7e5274e175da73fe53057a1c486789e338c'
import numpy as np
from threadpoolctl import threadpool_info
assert all(t['num_threads']==1 for t in threadpool_info())
old=read(inputs['old_qualification']['path']);assert old['completed'] and not old['response_qualified']
ref=read(inputs['parent_record']['path'])['records'][0]
candidate=read(inputs['candidate_record']['path'])
assert candidate['rank']==ref['rank']==35 and candidate['label']=='plus' and candidate['completed']
assert ref['includes_initialization'] and candidate['trust_envelope_passed'] and candidate['numeric_cache_retired']
assert sha(candidate['prediction']['path'])==candidate['prediction']['sha256']
assert sha(ref['output'])==ref['output_sha256']
yp=np.load(ref['output'],allow_pickle=False);yc=np.load(candidate['prediction']['path'],allow_pickle=False)
assert yp.dtype==yc.dtype==np.float32 and yp.shape==yc.shape==(16,1024)
assert np.isfinite(yp).all() and np.isfinite(yc).all() and (yp>=0).all() and (yc>=0).all()
assert yp[:2].tobytes()==yc[:2].tobytes()
with np.load(inputs['truth']['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
assert truth.shape==(1024,) and np.isfinite(truth).all()
records={}
for name,y,F in [('parent',yp,ref['flops']),('candidate',yc,candidate['complete_flops'])]:
    mse=float(np.mean((y[-1].astype(np.float64)-truth)**2));mult=max(.1,F/2**41)
    records[name]=dict(raw=mse,complete_cold_flops=F,multiplier=mult,adjusted=mse*mult)
ratio=records['candidate']['adjusted']/records['parent']['adjusted']
result=dict(completed=True,candidate_sha256=inputs['candidate']['sha256'],rank=35,
    records=records,raw_ratio=records['candidate']['raw']/records['parent']['raw'],adjusted_ratio=ratio,
    gate_threshold=.98,gate_passed=bool(ratio<=.98),read_other_probe_outputs=False,
    previous_response_gate_unchanged_failed=True,no_scalar_fit=True,new_model_calls=0,
    reused_proposal_training_network=True,independent_generalization=False,
    ordinary1B_reference_not_NC14=True,official_score=False,
    source_seal_sha256=sha(HERE/'sources_sealed.json'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
