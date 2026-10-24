"""Wait for successful capture, then run the two fixed fits and assessment serially."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;P2=HERE.parent.parent
guard=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
seal=json.loads((HERE/'pre_fit_source_seal.json').read_text())
for name,digest in seal['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256((HERE/'assess.py').read_bytes()).hexdigest()==json.loads((HERE/'launch_state.json').read_text())['assessment_sha256']
started=time.monotonic()
while not (HERE/'capture_complete.json').exists():
    assert time.monotonic()-started<6300,'capture completion timeout; no fit was started'
    for f in HERE.glob('capture_*_guard.json'):
        rec=json.loads(f.read_text())
        assert rec['completed'] and rec['whole_group_retired'] and rec['max_swap_growth_bytes']==0,(str(f),rec)
    time.sleep(1)
capture=json.loads((HERE/'capture_complete.json').read_text())
assert capture['all_groups_retired'] and capture['swap_growth_bytes']==0
assert len(capture['rows'])==160
for tag,runtime,script,args,seconds in [
    ('fit_graph','/Users/matt/anaconda3/bin/python','fit.py',['graph'],1800),
    ('fit_control','/Users/matt/anaconda3/bin/python','fit.py',['control'],1800),
    ('assessment',sys.executable,'assess.py',[],120)]:
    record=HERE/f'{tag}_guard.json';assert not record.exists()
    print(json.dumps(dict(event='stage_start',stage=tag)),flush=True)
    with (HERE/f'{tag}.log').open('x') as log:
        cp=subprocess.run([sys.executable,str(guard),'--record',str(record),'--seconds',str(seconds),runtime,str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT)
    result=json.loads(record.read_text())
    assert cp.returncode==0 and result['completed'] and result['whole_group_retired'] and result['max_swap_growth_bytes']==0,(tag,result)
    print(json.dumps(dict(event='stage_complete',stage=tag,elapsed_s=result['elapsed_s'],max_rss_bytes=result['max_rss_bytes'])),flush=True)
assessment=json.loads((HERE/'assessment.json').read_text())
with (HERE/'pipeline_complete.json').open('x') as f:
    json.dump(dict(elapsed_s=time.monotonic()-started,assessment_sha256=hashlib.sha256((HERE/'assessment.json').read_bytes()).hexdigest(),
                   all_groups_retired=True,passes_information_gate=assessment['passes_information_gate']),f,indent=2)
print(json.dumps(dict(event='pipeline_complete',passes_information_gate=assessment['passes_information_gate'],graph_vs_bank=assessment['graph_vs_bank'],control_vs_bank=assessment['control_vs_bank'])),flush=True)
