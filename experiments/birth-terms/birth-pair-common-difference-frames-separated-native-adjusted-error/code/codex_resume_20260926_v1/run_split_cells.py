"""Serial fresh-process screens avoid Torch's retained per-graph allocations."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
GUARD = P2 / 'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
ranks = [r for r in inputs['train'] if r not in (35, 100, 200)][::6][:4]
arms = ['l10', 'l10:su7=320,su10=256,su13=256', 'l10:su7=320,su10=192,su13=192']
for rank in ranks:
    for k, arm in enumerate(arms):
        tag = f'cell_{rank}_a{k}'
        with (HERE / (tag + '.log')).open('x') as log:
            command = [sys.executable, str(GUARD), '--record', str(HERE / (tag + '_guard.json')), '--seconds', '180',
                       '/Users/matt/anaconda3/bin/python', str(HERE / 'split_screen_s29.py'), str(rank), '1', tag, arm]
            p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(rank=rank, arm=arm, returncode=p.returncode)), flush=True)
        if p.returncode:
            raise SystemExit(p.returncode)
print('ALL_CELLS_COMPLETE', flush=True)
