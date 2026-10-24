"""Twenty-four public target-free tapes, reuse only verified complete rows."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

here=Path(__file__).resolve().parent
out=here/'late_transport_pilot_rows_v1.json'
assert not out.exists()
assert json.loads((here/'current_transport_late_corner_fidelity_v1.json').read_text())['fidelity_pass']
start=time.time()
def one(rank):
    folder=here/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
    if folder.exists():
        r=json.loads((folder/'result.json').read_text())
        assert r['rank']==rank and r['target_access'] is False
        assert all((folder/f'{key}.npy').exists() for key in ('pre_cov','k21','k4','mu','host','sigma','x','state_prev','delta','alive','weights','baseline'))
        return dict(rank=rank,reused=True,receipt=r)
    run=subprocess.run(['nice','-n','10','/Users/matt/arc-white-box-estimation/.venv/bin/python','-u',
                        str(here/'generate_current_transport_shadow_v2.py'),'--rank',str(rank)],cwd=here,capture_output=True,text=True)
    print(run.stdout,end='',flush=True)
    assert run.returncode==0,(rank,run.stderr)
    return dict(rank=rank,reused=False,receipt=json.loads((folder/'result.json').read_text()))
with ThreadPoolExecutor(max_workers=3) as pool:
    records=list(pool.map(one,range(35,59)))
out.write_text(json.dumps(dict(n=24,records=records,wall_s=time.time()-start),indent=2)+'\n')
print('LATE_TRANSPORT_ROWS_COMPLETE',flush=True)
