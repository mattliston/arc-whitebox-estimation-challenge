"""Exact control and independent scalar-fsum score comparison."""
from pathlib import Path
import json,hashlib,math,sys
import numpy as np
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
read=lambda p:json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
for x in read(HERE/'build.json')['sources']:assert pin(x['path'])==x
for x in read(HERE/'endpoint_seal.json')['sources']:assert pin(x['path'])==x
baseline=H2/'both_tail_train2_v6';disabled=H2/'cov_supplier_disabled_train2_v1'
brows=[json.loads(x) for x in (baseline/'rows.jsonl').read_text().splitlines()]
drows=[json.loads(x) for x in (disabled/'rows.jsonl').read_text().splitlines()]
assert [x['rank'] for x in brows]==[x['rank'] for x in drows]==[35,100]
assert read(disabled/'manifest.json')['sha256']==pin(HERE/'disabled.py')['sha256']
for b,d in zip(brows,drows):
    assert b['flops']==d['flops'] and d['cf']['calls']==0
    assert np.load(baseline/f"pred_{b['rank']}.npy").tobytes()==np.load(disabled/f"pred_{b['rank']}.npy").tobytes()
g=read(HERE/'disabled_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
if sys.argv[1]=='disabled':
    with (HERE/'disabled_qualified.json').open('x') as f:json.dump(dict(all16_bytes_exact=True,bills_exact=True,ranks=[35,100],candidate=pin(HERE/'disabled.py')),f,indent=2)
    print('DISABLED_ALL16_AND_BILLS_EXACT',flush=True);sys.exit(0)
assert read(HERE/'disabled_qualified.json')['all16_bytes_exact']
enabled=H2/'cov_supplier_enabled_train2_v1'
erows=[json.loads(x) for x in (enabled/'rows.jsonl').read_text().splitlines()]
assert [x['rank'] for x in erows]==[35,100]
assert read(enabled/'manifest.json')['sha256']==pin(HERE/'candidate.py')['sha256']
g=read(HERE/'enabled_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
rows=[]
for j,(b,e) in enumerate(zip(brows,erows)):
    rank=b['rank'];base=np.load(baseline/f'pred_{rank}.npy');new=np.load(enabled/f'pred_{rank}.npy')
    assert new[:9].tobytes()==base[:9].tobytes() and e['cf']['calls']==7*(j+1)
    assert np.isfinite(new).all() and np.min(new)>=0
    truth=np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype('float64')
    mse=math.fsum((float(a)-float(t))**2 for a,t in zip(new[15],truth))/1024
    bmse=math.fsum((float(a)-float(t))**2 for a,t in zip(base[15],truth))/1024
    assert math.isclose(mse,e['mse'],rel_tol=5e-15) and math.isclose(bmse,b['mse'],rel_tol=5e-15)
    adj=mse*max(.1,e['flops']/2**41);badj=bmse*max(.1,b['flops']/2**41)
    assert math.isclose(adj,e['adjusted'],rel_tol=5e-15)
    rows.append(dict(rank=rank,raw_mse=mse,parent_mse=bmse,flops=e['flops'],parent_flops=b['flops'],
        adjusted=adj,parent_adjusted=badj,raw_ratio=mse/bmse,bill_ratio=e['flops']/b['flops'],adjusted_ratio=adj/badj,first9_exact=True))
raw=math.fsum(r['raw_mse'] for r in rows)/math.fsum(r['parent_mse'] for r in rows)
bill=sum(r['flops'] for r in rows)/sum(r['parent_flops'] for r in rows)
adj=math.fsum(r['adjusted'] for r in rows)/math.fsum(r['parent_adjusted'] for r in rows)
gates=dict(adjusted=adj<=.99,raw=raw<=1,both_wins=all(r['adjusted_ratio']<1 for r in rows))
result=dict(rows=rows,pooled_raw_ratio=raw,pooled_bill_ratio=bill,pooled_adjusted_ratio=adj,
    gates=gates,passed=all(gates.values()),candidate=pin(HERE/'candidate.py'),
    scalar_replay_passed=True,model_refitted=False,original_observer_failure_unchanged=True)
with (HERE/'result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
