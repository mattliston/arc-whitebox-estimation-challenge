from pathlib import Path
import hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path,digest in read(HERE/'seal_v2.json')['sources'].items():assert sha(Path(path))==digest
assert read(HERE/'disabled_parity.json')['passed'] and read(HERE/'unit_v2_result.json')['passed']
g=read(HERE/'endpoint_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
base={r['rank']:r for r in map(json.loads,(H2/'both_tail_train2_v6/rows.jsonl').read_text().splitlines())}
got=[json.loads(x) for x in (H2/'fourth_matrix_train2_v1/rows.jsonl').read_text().splitlines()]
assert [r['rank'] for r in got]==[35,100];rows=[]
for row in got:
    rank=row['rank'];bp=H2/'both_tail_train2_v6'/f'pred_{rank}.npy';pp=H2/'fourth_matrix_train2_v1'/f'pred_{rank}.npy'
    y=np.load(pp);b=np.load(bp)
    assert y.shape==(16,1024) and np.isfinite(y).all() and y[:8].tobytes()==b[:8].tobytes()
    tp=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
    with np.load(tp) as z:t=z['mc_post'][15].astype(np.float64)
    mse=math.fsum((float(u)-float(v))**2 for u,v in zip(y[15],t))/1024
    bmse=math.fsum((float(u)-float(v))**2 for u,v in zip(b[15],t))/1024
    assert abs(mse-row['mse'])<1e-20 and abs(bmse-base[rank]['mse'])<1e-20
    adj=mse*max(.1,row['flops']/2**41);badj=bmse*max(.1,base[rank]['flops']/2**41)
    assert abs(adj-row['adjusted'])<1e-22
    rows.append(dict(rank=rank,mse=mse,flops=row['flops'],adjusted=adj,parent_mse=bmse,parent_flops=base[rank]['flops'],
                     parent_adjusted=badj,raw_ratio=mse/bmse,adjusted_ratio=adj/badj,first8_bytes_exact=True,
                     prediction_sha256=sha(pp),truth_sha256=sha(tp)))
raw=math.fsum(r['mse'] for r in rows)/math.fsum(r['parent_mse'] for r in rows)
adj=math.fsum(r['adjusted'] for r in rows)/math.fsum(r['parent_adjusted'] for r in rows)
bill=math.fsum(r['flops'] for r in rows)/math.fsum(r['parent_flops'] for r in rows)
gates=dict(two_wins=all(r['adjusted_ratio']<1 for r in rows),pooled_adjusted=adj<=.98,raw_each=all(r['raw_ratio']<=1.05 for r in rows))
result=dict(passed=all(gates.values()),gates=gates,rows=rows,pooled_raw_ratio=raw,pooled_adjusted_ratio=adj,pooled_bill_ratio=bill,
            candidate_sha256=sha(HERE/'candidate_v2.py'),model_sha256=read(HERE/'build_v2.json')['model']['sha256'],
            original_proxy_gate_passed=False,explicit_separate_endpoint=True,valid40_accessed=False,submitted=False,
            all_groups_retired=True,added_swap_bytes=0)
with (HERE/'endpoint_result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
