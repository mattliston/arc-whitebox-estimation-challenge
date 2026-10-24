"""Stdlib-only serial acquisition; no target decoding or unguarded numerics."""
import subprocess,sys,time
from common import HERE,D,P2,TRAIN,CHECK,sha,read,seal_check,write
seal_check('acquisition_sources_sealed.json');started=time.monotonic()
q=read(HERE/'neural_qualification/result.json');g=read(HERE/'neural_qualification_watch.json')
assert q['passed'] and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
root=HERE/'captures';root.mkdir(exist_ok=False)
first=read(HERE/'capture_qualification/result.json')
rows=[dict(ordinal=0,rank=35,record=first['records'][0],parent_P=first['parent_P'],watch=str(HERE/'capture_qualification_watch.json'),completion=str(HERE/'capture_qualification/result.json'),completion_sha256=sha(HERE/'capture_qualification/result.json'))]
runtime=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
for ordinal,rank in enumerate(TRAIN[1:]+CHECK,1):
    assert time.monotonic()-started<2300,'serial acquisition deadline'
    watch=root/f'rank{rank:04d}_watch.json';log=root/f'rank{rank:04d}.log'
    with log.open('x') as stream:
        run=subprocess.run([sys.executable,str(D/'run_guarded.py'),'--record',str(watch),'--seconds','100',
            str(runtime),'-B','-W','ignore',str(HERE/'capture.py'),'--rank',str(rank)],
            cwd=D,stdout=stream,stderr=subprocess.STDOUT,timeout=110)
    assert run.returncode==0,rank
    guard=read(watch);assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0,rank
    receipt=root/f'rank{rank:04d}'/'result.json';r=read(receipt)
    assert r['completed'] and not r['truth_accessed'] and len(r['records'])==1
    rec=r['records'][0]
    assert rec['rank']==rank and rec['includes_initialization'] and rec['all16_parent_bytes_and_bill_exact']
    assert sha(rec['path'])==rec['sha256'] and r['parent_P']['sha256']==first['parent_P']['sha256']
    rows.append(dict(ordinal=ordinal,rank=rank,record=rec,parent_P=r['parent_P'],watch=str(watch),completion=str(receipt),completion_sha256=sha(receipt)))
    write(root/f'progress_{ordinal:02d}.json',dict(completed_rows=len(rows),last_rank=rank,elapsed_s=time.monotonic()-started))
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<220_000_000
    print(f'RETIRED rank={rank} rows={len(rows)}/24 elapsed={time.monotonic()-started:.1f}s',flush=True)
assert [r['rank'] for r in rows]==TRAIN+CHECK
write(root/'manifest.json',dict(completed=True,truth_accessed=False,train=TRAIN,check=CHECK,rows=rows,
    source_seal_sha256=sha(HERE/'acquisition_sources_sealed.json'),capture_source_sha256=sha(HERE/'capture.py'),
    elapsed_s=time.monotonic()-started,new_full_parent_calls=23,reused_qualified_cold_rows=1))
print('ALL24_COMPLETE_AND_SEALED',flush=True)
