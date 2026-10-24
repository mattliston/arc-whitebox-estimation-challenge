import os,json,ast,hashlib,collections,time,resource
from pathlib import Path
assert os.getpriority(os.PRIO_PROCESS,0)==10
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'lead/runs/astra_smirnov_unseen_shapes_v1';D.mkdir(exist_ok=True);S=O/'lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py';T=ast.parse(S.read_text());plans=next(ast.literal_eval(n.value)for n in T.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='_ASTRA_SM40_PLANS');del T
current=set()
for p in(O/'lead/runs/astra_prefix_row_energy22_science_v1').glob('run_rpe_*_s*_v1/native_*_ON.json'):current.add(json.loads(p.read_text())['rows'][0]['seed'])
rows={};scanned=0;withstats=0
for line in Path('/private/tmp/astra_native_json_inventory.txt').read_text().splitlines():
 p=Path(line)
 if '_guard' in p.name:continue
 scanned+=1
 try:j=json.loads(p.read_text())
 except Exception:continue
 for r in j.get('rows',[]):
  seed=r.get('seed');reads=r.get('stats',{}).get('_RTT_STATS',{}).get('reads',[])
  if not reads or seed in current:continue
  withstats+=1
  if len(reads)!=14:continue
  if any(q.get('old_rank')!=(0 if q['layer']<7 else 320 if 10<=q['layer']<=12 else 384)for q in reads):continue
  shapes=[tuple(q[k]for k in ('m','k','c15','old_rank'))for q in reads]
  if any(m%32 or n%32 or k%64 for m,k,n,rank in shapes):continue
  if seed not in rows or p.stat().st_mtime>rows[seed]['mtime']:
   rows[seed]=dict(seed=seed,path=str(p),mtime=p.stat().st_mtime,flops=r.get('flops'),shapes=shapes)
counts=collections.Counter(tuple(s)for r in rows.values()for s in r['shapes']);novel={k:v for k,v in counts.items()if k not in plans}
out=dict(scope='Cost/geometry only; most recent compatible-rank read trace per seed excluding all24 dispatch-selection seeds. No MSE fields consumed. Saved sources may differ in gating and arithmetic; this is coverage evidence, not a current accuracy estimate.',source_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),scanned=scanned,cert_rows_with_stats=withstats,current_excluded_seeds=sorted(current),networks=list(rows.values()),unique_shapes=len(counts),novel_shapes=[dict(shape=k,count=v)for k,v in novel.items()],novel_calls=sum(novel.values()),calls=sum(counts.values()),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(D/'CENSUS_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ('networks','novel_shapes','current_excluded_seeds')}));print('seeds',sorted(rows));print('newunique',len(novel))
