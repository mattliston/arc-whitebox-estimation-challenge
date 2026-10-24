"""Exactly24 branch-specific on-policy final rows, three CPU workers."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--branch',required=True,choices=('control','positive'));a=ap.parse_args()
OUT=HERE/f'positive_path_pilot_final_rows_{a.branch}_v1.json';assert not OUT.exists()
PY='/Users/matt/arc-white-box-estimation/.venv/bin/python';start=time.time();records=[]
def run(rank):
    p=subprocess.run([PY,'-u',str(HERE/'generate_positive_path_pilot_final_row_v1.py'),'--branch',a.branch,'--rank',str(rank)],cwd=HERE,capture_output=True,text=True)
    if p.returncode:raise RuntimeError((rank,p.stdout,p.stderr))
    return json.loads((HERE/f'rows_positive_path_pilot_final_{a.branch}_v1'/f'row_{rank:04d}/result.json').read_text())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(run,r) for r in range(35,59)]
    for i,f in enumerate(concurrent.futures.as_completed(futures),1):
        r=f.result();records.append(r)
        if i%3==0:print(json.dumps(dict(branch=a.branch,rows=i,rank=r['rank'],wall_s=time.time()-start)),flush=True)
record=dict(branch=a.branch,records=records,wall_s=time.time()-start)
OUT.write_text(json.dumps(record,indent=2)+'\n')
