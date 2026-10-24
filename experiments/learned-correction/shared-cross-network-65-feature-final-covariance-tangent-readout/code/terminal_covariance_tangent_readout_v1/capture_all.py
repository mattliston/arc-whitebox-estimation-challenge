"""Stdlib-only serial launcher; each numerical row has its own bounded group."""
import subprocess,sys,time
from common import HERE,D,P2,TRAIN,sha,read,seal_check,write
seal_check();started=time.monotonic()
g=read(HERE/'qualification_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
q=read(HERE/'qualification/result.json');assert q['completed'] and q['feature_repeat_exact']
root=HERE/'train_capture';root.mkdir(exist_ok=False)
rows=[dict(ordinal=0,rank=35,record=q['records'][0],watch=str(HERE/'qualification_watch.json'),completion=str(HERE/'qualification/result.json'),completion_sha256=sha(HERE/'qualification/result.json'))]
runtime=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
for ordinal,rank in enumerate(TRAIN[1:],1):
    assert time.monotonic()-started<1800,'serial acquisition deadline'
    watch=root/f'rank{rank:04d}_watch.json';log=root/f'rank{rank:04d}.log'
    with log.open('x') as stream:
        run=subprocess.run([sys.executable,str(D/'run_guarded.py'),'--record',str(watch),'--seconds','100',str(runtime),str(HERE/'capture.py'),str(rank)],cwd=D,stdout=stream,stderr=subprocess.STDOUT,timeout=110)
    assert run.returncode==0,rank
    guard=read(watch);assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0,rank
    receipt=root/f'rank{rank:04d}'/'result.json';r=read(receipt);assert r['completed'] and not r['truth_accessed']
    rec=r['record'];assert rec['rank']==rank and rec['parent_all16_and_bill_exact'] and sha(rec['capture'])==rec['capture_sha256']
    rows.append(dict(ordinal=ordinal,rank=rank,record=rec,watch=str(watch),completion=str(receipt),completion_sha256=sha(receipt)))
    write(root/f'progress_{ordinal:02d}.json',dict(completed_rows=len(rows),last_rank=rank,elapsed_s=time.monotonic()-started))
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) < 60_000_000
    print(f'RETIRED rank={rank} rows={len(rows)}/20 elapsed={time.monotonic()-started:.1f}s',flush=True)
assert [r['rank'] for r in rows]==TRAIN
write(root/'manifest.json',dict(completed=True,truth_accessed=False,train=TRAIN,rows=rows,source_seal_sha256=sha(HERE/'sources_sealed.json'),capture_source_sha256=sha(HERE/'capture.py'),feature_source_sha256=sha(HERE/'feature.py'),elapsed_s=time.monotonic()-started,new_full_model_calls=19,reused_qualified_cold_rows=1))
print('ALL20_COMPLETE_AND_SEALED',flush=True)
