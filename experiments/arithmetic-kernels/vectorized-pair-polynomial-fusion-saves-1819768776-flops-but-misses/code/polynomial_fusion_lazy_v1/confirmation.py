"""Eight fixed candidate cold calls joined to authenticated Lazy cold calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
RUNTIME=D.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=read(HERE/'build.json')
assert read(HERE/'panel_score.json')['panel_gate_passed']
parent=read(D/'lazy_rotation_objective_v1/confirmation/completed.json')
assert parent['completed'] and parent['candidate_sha256']==build['parent_sha256']
NETS=[37,39,42,48,59,63,66,67]
assert [r['rank'] for r in parent['records']]==NETS
parents={r['rank']:r['arms']['candidate'] for r in parent['records']}
out=HERE/'confirmation';out.mkdir(exist_ok=False)
records=[]
for rank in NETS:
    assert sha(HERE/'candidate.py')==build['candidate_sha256']
    old=parents[rank]
    watch=read(old['watch'])
    assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes']==0
    assert read(old['result'])['candidate_sha256']==build['parent_sha256']
    target=out/f'candidate_{rank}'
    watch_path=out/f'candidate_{rank}_watch.json'
    command=[sys.executable,str(D/'run_guarded.py'),'--record',str(watch_path),'--seconds','100',
             str(RUNTIME),'-B','-W','ignore',str(D/'weight5_reuse_v1/native.py'),str(HERE/'candidate.py'),str(target),'--ranks',str(rank)]
    with (out/f'candidate_{rank}.log').open('x') as log:
        done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    assert done.returncode==0,(rank,done.returncode)
    watch=read(watch_path)
    assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes']==0
    receipt=read(target/'result.json')
    assert receipt['candidate_sha256']==build['candidate_sha256'] and not receipt['truth_accessed']
    call=receipt['records'][0]
    assert call['rank']==old['call']['rank']==rank and call['includes_initialization'] and old['call']['includes_initialization']
    assert call['weights_sha256']==old['call']['weights_sha256']
    for item in (call,old['call']):assert sha(item['output'])==item['output_sha256']
    assert call['weight5_cache']['completed'] and call['weight5_cache']['numeric_cache_retired']
    row=dict(rank=rank,arms={'candidate':dict(call=call,result=str(target/'result.json'),watch=str(watch_path)),'lazy':old})
    records.append(row)
    with (out/f'verified_{rank}.json').open('x') as f:json.dump(row,f,indent=2)
    print(json.dumps(dict(rank=rank,flops=call['flops'],saved_flops=old['call']['flops']-call['flops'],wall_s=call['wall_s'],peak_rss_bytes=watch['max_rss_bytes'])),flush=True)
with (out/'completed.json').open('x') as f:
    json.dump(dict(completed=True,records=records,new_calls=8,candidate_sha256=build['candidate_sha256'],
                   parent_sha256=build['parent_sha256'],truth_accessed=False,driver_sha256=sha(__file__)),f,indent=2)
