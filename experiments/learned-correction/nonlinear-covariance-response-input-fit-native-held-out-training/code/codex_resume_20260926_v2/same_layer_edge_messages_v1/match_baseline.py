"""Match the baseline's worker order and cache history; no candidate or fit changes."""
from pathlib import Path
import json
import subprocess
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent;ROOT=P2.parents[3]
LIGHT=ROOT/'.venv/bin/python';NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
protocol=json.loads((HERE/'protocol.json').read_text())
for mode in ('message','control'):
    g=json.loads((HERE/f'check_{mode}_guard.json').read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
def run(name,seconds,command):
    print(json.dumps(dict(phase=name)),flush=True)
    with (HERE/f'{name}.log').open('x') as fp:
        subprocess.run([str(LIGHT),str(GUARD),'--record',str(HERE/f'{name}_guard.json'),'--seconds',str(seconds),*map(str,command)],cwd=H2,stdout=fp,stderr=subprocess.STDOUT,check=True)
    g=json.loads((HERE/f'{name}_guard.json').read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
run('matched_baseline',900,[NATIVE,H2/'native_eval.py',H2/'candidate_old_both_tail_trace_bounded_v6.py',
    ','.join(map(str,protocol['check_ranks'])),'same_layer_edge_baseline_check8_v1'])
run('assessment_matched',60,[NATIVE,HERE/'assess.py','--matched-baseline'])
