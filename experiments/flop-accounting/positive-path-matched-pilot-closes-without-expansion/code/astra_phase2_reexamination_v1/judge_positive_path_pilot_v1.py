"""Fixed four-holdout pooled gate, three nice-inheriting CPU card workers."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;OUT=HERE/'positive_path_pilot_verdict_v1.json';assert not OUT.exists()
for b in ('control','positive'):assert (HERE/f'positive_path_pilot_final_training_{b}_v1.json').exists()
PY='/Users/matt/arc-white-box-estimation/.venv/bin/python';start=time.time();records=[]
def run(rank):
    p=subprocess.run([PY,'-u',str(HERE/'judge_positive_path_pilot_row_v1.py'),'--rank',str(rank)],cwd=HERE,capture_output=True,text=True)
    if p.returncode:raise RuntimeError((rank,p.stdout,p.stderr))
    return json.loads((HERE/f'positive_path_pilot_judge_r{rank}_v1.json').read_text())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(run,r) for r in (37,39,42,48)]
    for f in concurrent.futures.as_completed(futures):
        r=f.result();records.append(r);print(json.dumps(dict(rank=r['rank'],final_mse=r['final_mse'],positive_bank_ratio=r['positive_bank_ratio'],positive_control_ratio=r['positive_control_ratio'])),flush=True)
records.sort(key=lambda r:r['rank'])
means={k:sum(r['final_mse'][k] for r in records)/4 for k in ('bank','control','positive')}
rc=means['positive']/means['control'];rb=means['positive']/means['bank'];wins=sum(r['positive_bank_ratio']<1 for r in records)
passed=rc<=.90 and rb<=.92 and wins>=3
record=dict(records=records,pooled_final_mse=means,new_control_ratio=rc,new_bank_ratio=rb,bank_wins=wins,
            matched_pilot_gate=passed,wall_s=time.time()-start,
            qualification='Fixed20/4 two-stage matched pilot, final checkpoints only, actual NumPy closed-loop endpoints. No source/epoch/rank tuning. Not14-net/adjusted/official scoring.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='records'},indent=2),flush=True)
