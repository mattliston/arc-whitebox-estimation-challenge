import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
GUARD = P2 / 'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
# Fixed development ranks; shared with the preceding screen, never claimed pristine.
for rank in (36, 45, 52, 58):
    for kind, arm in [('nystrom', 'sched384-320-384'), ('eigh', 'sched320-256-320'), ('eigh', 'sched256-192-256')]:
        tag = f'spectral_{rank}_{kind}_{arm}'
        env = dict(os.environ, SPECTRAL_KIND=kind)
        with (HERE / (tag+'.log')).open('x') as log:
            cmd = [sys.executable, str(GUARD), '--record', str(HERE/(tag+'_guard.json')), '--seconds', '180',
                   '/Users/matt/anaconda3/bin/python', str(HERE/'spectral_screen_s29.py'), str(rank), '1', tag+'.jsonl', arm]
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(rank=rank, kind=kind, arm=arm, returncode=proc.returncode)), flush=True)
        if proc.returncode: raise SystemExit(proc.returncode)
print('ALL_CELLS_COMPLETE', flush=True)
