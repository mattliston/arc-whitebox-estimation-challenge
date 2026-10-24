"""Serial fixed continuation after the message fit: control, native CHECK8, assessment."""
from pathlib import Path
import json
import subprocess
import time
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
ROOT=P2.parents[3]
LIGHT=ROOT/'.venv/bin/python';TORCH=Path('/Users/matt/anaconda3/bin/python')
NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
assert LIGHT.exists() and NATIVE.exists() and GUARD.exists()
g=json.loads((HERE/'fit_message_guard.json').read_text())
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert not (HERE/'assessment.json').exists()
protocol=json.loads((HERE/'protocol.json').read_text())
def phase(name):
    state=dict(phase=name,updated_unix=time.time(),note='Progress marker only; consult guard receipts and process state before any restart.')
    (HERE/'remainder_state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
def guarded(name,seconds,argv,cwd=HERE):
    phase(name)
    receipt=HERE/f'{name}_guard.json'
    with (HERE/f'{name}.log').open('x') as fp:
        subprocess.run([str(LIGHT),str(GUARD),'--record',str(receipt),'--seconds',str(seconds),*map(str,argv)],cwd=cwd,stdout=fp,stderr=subprocess.STDOUT,check=True)
    g=json.loads(receipt.read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,name
guarded('fit_control',7200,[TORCH,HERE/'fit.py','control'])
for mode in ('message','control'):
    phase('build_'+mode)
    subprocess.run([str(LIGHT),str(HERE/'build_native.py'),mode,'terminal'],cwd=HERE,check=True)
for mode in ('message','control'):
    guarded('check_'+mode,900,[NATIVE,H2/'native_eval.py',H2/f'candidate_same_layer_edge_{mode}_terminal_v1.py',
        ','.join(map(str,protocol['check_ranks'])),f'same_layer_edge_{mode}_check8_v1'],cwd=H2)
guarded('assessment',60,[NATIVE,HERE/'assess.py'])
phase('fixed_pilot_complete')
