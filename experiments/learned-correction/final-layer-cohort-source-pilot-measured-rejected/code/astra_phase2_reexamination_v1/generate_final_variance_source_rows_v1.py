"""At most two low-priority two-thread workers; no fitting or targets."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('smoke','pilot'),required=True);a=ap.parse_args()
OUT=HERE/f'final_variance_source_rows_{a.stage}_v1.json';assert not OUT.exists()
if a.stage=='pilot':
    assert json.loads((HERE/'final_variance_source_rows_smoke_v1.json').read_text())['passed']
ranks=(35,100) if a.stage=='smoke' else tuple(range(35,59))
PY='/Users/matt/arc-white-box-estimation/.venv/bin/python'
start=time.time();records=[]
def one(rank):
    child=subprocess.run([PY,str(HERE/'generate_final_variance_source_row_v1.py'),'--rank',str(rank)],cwd=HERE,capture_output=True,text=True)
    if child.returncode:raise RuntimeError((rank,child.stdout,child.stderr))
    record=json.loads((HERE/'rows_final_variance_source_probe_v1'/f'row_{rank:04d}/result.json').read_text())
    assert record['passed'];return record
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for i,future in enumerate(concurrent.futures.as_completed([pool.submit(one,r) for r in ranks]),1):
        record=future.result();records.append(record)
        print(json.dumps(dict(stage=a.stage,completed=i,total=len(ranks),rank=record['rank'],wall_s=time.time()-start)),flush=True)
result=dict(stage=a.stage,records=records,passed=all(r['passed'] for r in records),
            wall_s=time.time()-start,qualification='Input-only rows; no label file opened or fitting started.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
