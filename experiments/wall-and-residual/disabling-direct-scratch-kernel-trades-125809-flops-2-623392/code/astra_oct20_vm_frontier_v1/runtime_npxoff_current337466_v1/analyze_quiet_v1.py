"""Read-only complete three-rank exact prediction/tariff/runtime report."""
import argparse,hashlib,importlib.util,json,statistics,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
ap=argparse.ArgumentParser();ap.add_argument('--candidate',choices=['npxoff'],required=True);ap.add_argument('--reuse-parent',action='store_true');args=ap.parse_args();assert args.reuse_parent, 'This screening protocol reuses its pinned recent parent control.'
plan=read(HERE/'QUIET_PLAN_v1.json')
for p,h in plan['pins'].items():assert sha(p)==h,('pin changed',p)
hp=next(Path(p) for p in plan['pins'] if Path(p).name=='prepare_runtime_pair_parity_v3.py')
spec=importlib.util.spec_from_file_location('runtime_proof_only',hp);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
results={};proof={}
for key in ['control',args.candidate]:
 d=Path(plan['original_parent_quiet']) if key=='control' and args.reuse_parent else Path(plan['output'])/key
 guard=h.valid_guard(d/'GUARD_v1.json',original=True);m=read(d/'manifest.json');launcher=read(d/'launcher.json')
 assert m['sha256']==launcher['sha256']==plan['sources'][key]['sha256']
 assert m['ranks']==[35,150,200] and m['client_flopscope']=='0.12.3'
 assert m['diagnostic_harness_sha256']==plan['pins'][str(h.B/'boundary_client_dual_v3.py')]
 assert m['thread_helper_sha256']==plan['pins'][str(h.B/'thread_counts_v3.py')]
 assert launcher['client_exit']==launcher['server_exit']==0 and launcher['handle_cap']==100000
 rr=h.rows(d/'rows.jsonl');assert [r['rank'] for r in rr]==[35,150,200]
 for r,ref in zip(rr,plan['references']):
  assert sha(ref['prediction'])==ref['prediction_sha256']
  assert h.valid_row(r,d/f"pred_{r['rank']}.bin")==Path(ref['prediction']).read_bytes(),('not exact',key,r['rank'])
  assert r['compile_stats']['missing']==r['compile_stats']['capacity_skipped']==0
  if key=='control':assert r['flops']==ref['row']['flops']
 results[key]=rr;proof[key]=dict(path=str(d),source_sha256=m['sha256'],rows_sha256=sha(d/'rows.jsonl'),guard_sha256=sha(d/'GUARD_v1.json'),setup=m['setup'],full_three_prediction_bytes_exact=True,whole_group_retired=True)
rows=[]
fields=['wall_s','backend_s','thread_residual_instructions','requests','flops']
for base,cand in zip(results['control'],results[args.candidate]):
 assert cand['flops']>base['flops'] and cand['requests']<base['requests']
 rows.append(dict(rank=base['rank'],**{k+'_pct':100*(cand[k]/base[k]-1) for k in fields},flop_delta=cand['flops']-base['flops'],request_delta=cand['requests']-base['requests'],control={k:base[k] for k in fields},candidate={k:cand[k] for k in fields}))
mean={k:statistics.mean(r[k] for r in rows) for k in rows[0] if k.endswith('_pct')}
r=dict(utc=h.utc(),status='WHOLE_THREE_PREDICTION_PARITY_AND_MEASURED_TARIFF',candidate=args.candidate,plan_sha256=sha(HERE/'QUIET_PLAN_v1.json'),proof=proof,rows=rows,mean_changes_pct=mean,reused_control=args.reuse_parent,no_full32_transfer=True,no_release_or_fire=True,limitation='REUSED EARLIER CONTROL: runtime is descriptive only, not causal paired evidence. Exact own predictions/tariff/guards remain checked. No hosted residual guarantee/full32 transfer/fire.')
target=HERE/f'QUIET_RESULT_{args.candidate}_v1.json'
with target.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(status=r['status'],mean=mean,path=str(target))))
