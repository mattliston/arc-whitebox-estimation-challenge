"""Serial, frozen FIT2 response diagnostic after the matched pilot assessment."""
from pathlib import Path
import json
import subprocess
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent;ROOT=P2.parents[3]
LIGHT=ROOT/'.venv/bin/python';NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
TORCH=Path('/Users/matt/anaconda3/bin/python');GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
assessment=json.loads((HERE/'assessment_matched.json').read_text());assert not assessment['passes_pilot']
g=json.loads((HERE/'assessment_matched_guard.json').read_text());assert g['completed'] and g['whole_group_retired']
protocol=json.loads((HERE/'fit_transfer_protocol.json').read_text())
def run(name,seconds,command,cwd=H2):
    print(json.dumps(dict(phase=name)),flush=True)
    with (HERE/f'{name}.log').open('x') as fp:
        subprocess.run([str(LIGHT),str(GUARD),'--record',str(HERE/f'{name}_guard.json'),'--seconds',str(seconds),*map(str,command)],cwd=cwd,stdout=fp,stderr=subprocess.STDOUT,check=True)
    g=json.loads((HERE/f'{name}_guard.json').read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
for mode in protocol['arms']:
    run('transfer_native_'+mode,180,[NATIVE,H2/'native_eval.py',H2/f'candidate_same_layer_edge_{mode}_terminal_v1.py',
        ','.join(map(str,protocol['ranks'])),f'same_layer_edge_{mode}_fit2_transfer_v1'])
run('fit_transfer',120,[TORCH,HERE/'fit_transfer.py'],cwd=HERE)
