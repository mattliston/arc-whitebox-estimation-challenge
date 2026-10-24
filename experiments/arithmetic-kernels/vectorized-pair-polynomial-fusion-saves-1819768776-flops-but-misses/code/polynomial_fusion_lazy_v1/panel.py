"""Stdlib-only driver: twelve new serial cold calls plus two authenticated saved calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
SOURCE=HERE
PARENT=D/'lazy_rotation_objective_v1'
RUNTIME=D.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
NETS=(35,100,150,200,250,300,400,500,3016500,3016501,3016502,3016503,3016504,3016505)
EXPECTED='29e9211b4136bde7a74c9f554e697bd0e7e0c77e69326e5c365bafedfa9d8b07'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert read(HERE/'dev_score.json')['development_gate_passed']
parents={r['rank']:r for r in read(PARENT/'panel/completed.json')['records']}
out=HERE/'panel'
out.mkdir(exist_ok=False)
records=[]
for rank in NETS:
    assert sha(HERE/'candidate.py')==EXPECTED
    if rank in (35,100):
        target=SOURCE/('dev' if rank==35 else 'cold100')
        watch_path=SOURCE/('dev_watch.json' if rank==35 else 'cold100_watch.json')
    else:
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
    assert receipt['candidate_sha256']==EXPECTED and not receipt['truth_accessed']
    call=receipt['records'][0]
    old=parents[rank]['call']
    assert call['rank']==rank and call['includes_initialization']
    assert call['weights_sha256']==old['weights_sha256']
    for item in (call,old):
        assert sha(item['output'])==item['output_sha256']
    assert call['weight5_cache']['completed'] and call['weight5_cache']['numeric_cache_retired']
    row=dict(rank=rank,result=str(target/'result.json'),watch=str(watch_path),call=call,
             parent_result=parents[rank]['result'],saved_flops=old['flops']-call['flops'],
             flop_ratio=call['flops']/old['flops'])
    records.append(row)
    with (out/f'verified_{rank}.json').open('x') as f:json.dump(row,f,indent=2)
    print(json.dumps(dict(rank=rank,flops=call['flops'],saved_flops=row['saved_flops'],
                          wall_s=call['wall_s'],peak_rss_bytes=watch['max_rss_bytes'])),flush=True)
with (out/'completed.json').open('x') as f:
    json.dump(dict(completed=True,records=records,new_calls=12,candidate_sha256=EXPECTED,
                   truth_accessed=False,driver_sha256=sha(__file__)),f,indent=2)
