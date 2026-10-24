"""Serial groups; never restart a partial group automatically."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
HERE=Path(__file__).resolve().parent;P2=HERE.parent.parent;ROOT=P2.parents[3]
protocol=json.loads((HERE/'protocol.json').read_text())
guard=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
runtime=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
smoke=json.loads((HERE/'capture_00_guard.json').read_text())
assert smoke['completed'] and smoke['whole_group_retired'] and smoke['max_swap_growth_bytes']==0
assert json.loads((HERE/'capture_00'/'complete.json').read_text())['independent_controls_exact']==2
for g in range(1,len(protocol['capture_groups'])):
    record=HERE/f'capture_{g:02d}_guard.json'
    assert not record.exists() and not (HERE/f'capture_{g:02d}').exists()
    with (HERE/f'capture_{g:02d}.log').open('x') as log:
        cp=subprocess.run([sys.executable,str(guard),'--record',str(record),'--seconds','900',str(runtime),str(HERE/'capture.py'),str(g)],stdout=log,stderr=subprocess.STDOUT)
    rec=json.loads(record.read_text())
    assert cp.returncode==0 and rec['completed'] and rec['whole_group_retired'] and rec['max_swap_growth_bytes']==0,(g,rec)
    done=json.loads((HERE/f'capture_{g:02d}'/'complete.json').read_text())
    print(json.dumps(dict(group=g,rows=len(done['records']),max_rss_bytes=rec['max_rss_bytes'],elapsed_s=rec['elapsed_s'])),flush=True)
rows=[];guards=[]
for g in range(len(protocol['capture_groups'])):
    rec=json.loads((HERE/f'capture_{g:02d}_guard.json').read_text())
    assert rec['completed'] and rec['whole_group_retired'] and rec['max_swap_growth_bytes']==0
    done=json.loads((HERE/f'capture_{g:02d}'/'complete.json').read_text())
    assert done['protocol_sha256']==hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    assert [r['rank'] for r in done['records']]==protocol['capture_groups'][g]
    rows.extend(done['records']);guards.append(rec)
assert len(rows)==160 and len({r['rank'] for r in rows})==160
assert sum(r['bytes'] for r in rows)<protocol['capture_bytes_cap']
out=dict(rows=rows,targets_accessed=False,all_groups_retired=True,bytes=sum(r['bytes'] for r in rows),
         max_rss_bytes=max(g['max_rss_bytes'] for g in guards),swap_growth_bytes=0,
         graph_sha256=hashlib.sha256((HERE/'graph.py').read_bytes()).hexdigest(),
         capture_sha256=hashlib.sha256((HERE/'capture.py').read_bytes()).hexdigest(),
         protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest())
with (HERE/'capture_complete.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(dict(event='all_captures_complete',rows=len(rows),bytes=out['bytes'])),flush=True)
