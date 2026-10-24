import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
P2=HERE.parent
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
for rank in (36,45,52,58):
    for arm in ('sched384-320-384','fixed384','sched448-384-448'):
        tag=f'interpolation_{rank}_{arm}'
        with (HERE/(tag+'.log')).open('x') as log:
            proc=subprocess.run([sys.executable,str(GUARD),'--record',str(HERE/(tag+'_guard.json')),'--seconds','180',
                '/Users/matt/anaconda3/bin/python',str(HERE/'interpolation_screen_s29.py'),str(rank),'1',tag+'.jsonl',arm],stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(rank=rank,arm=arm,returncode=proc.returncode)),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)
print('ALL_CELLS_COMPLETE',flush=True)
