"""Sequential stdlib launcher; exactly one guarded numerical child at a time."""
import subprocess,sys
from common import D,HERE,FIT,read
ranks=tuple(map(int,sys.argv[1].split(',')))
assert ranks in ((35,),(36,38,40))
if ranks!=(35,):assert read(HERE/'qualification35.json')['response_qualified']
runtime=D.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
for rank in ranks:
    labels=('plus','minus','halfplus','halfminus','repeat') if rank==35 else ('plus','minus','halfplus','halfminus')
    for label in labels:
        tag=f'native_rank{rank:04d}_{label}'
        log=HERE/(tag+'.log');watch=HERE/(tag+'_watch.json')
        assert not log.exists() and not watch.exists()
        with log.open('x') as stream:
            child=subprocess.run([sys.executable,str(D/'run_guarded.py'),'--record',str(watch),
                '--seconds','90',str(runtime),str(HERE/'native.py'),str(rank),label],
                cwd=D,stdout=stream,stderr=subprocess.STDOUT)
        guard=read(watch)
        print(dict(rank=rank,label=label,exit_code=child.returncode,guard=guard),flush=True)
        assert child.returncode==0 and guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
print('ROSTER_COMPLETE',flush=True)
