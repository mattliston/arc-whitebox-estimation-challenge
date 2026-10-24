"""Two nice-10 workers for the fixed24-net complement-information gate."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import time

here=Path(__file__).resolve().parent
out=here/'relu_complement_probe_verdict_v1.json'
assert not out.exists()
start=time.time()
def one(rank):
    run=subprocess.run(['nice','-n','10','/Users/matt/arc-white-box-estimation/.venv/bin/python','-u',
        str(here/'generate_relu_complement_probe_v1.py'),'--rank',str(rank)],cwd=here,capture_output=True,text=True)
    print(run.stdout,end='',flush=True)
    assert run.returncode==0,(rank,run.stderr)
    return json.loads((here/'rows_relu_complement_probe_v1'/f'row_{rank:04d}'/'result.json').read_text())
with ThreadPoolExecutor(max_workers=2) as pool:
    records=list(pool.map(one,range(35,59)))
held=[r for r in records if r['rank'] in {37,39,42,48}]
base=statistics.mean(statistics.mean(r['layer_base_mse']) for r in held)
sym=statistics.mean(statistics.mean(r['layer_symmetric_mse']) for r in held)
improved=sum(statistics.mean(r['layer_symmetric_mse'])<statistics.mean(r['layer_base_mse']) for r in held)
receipt=dict(records=records,valid_nets=[37,39,42,48],base_mse=base,symmetric_mse=sym,
    ratio=sym/base,heldout_nets_improved=improved,information_gate_pass=sym<=.95*base and improved>=3,
    wall_s=time.time()-start,qualification='Fixed local teacher-forced complement read, not a recursive or board score.')
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='records'},indent=2),flush=True)
