"""Fixed four-holdout, actual closed-loop paired pilot verdict."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
OUT = HERE/'mean_response_transport_pilot_verdict_v1.json'
assert not OUT.exists()
assert (HERE/'mean_response_transport_pilot_final_training_v1.json').exists()
PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'
started = time.time()
def run(rank):
    p = subprocess.run([PY, '-u', str(HERE/'judge_mean_response_transport_pilot_row_v1.py'),
                        '--rank', str(rank)], cwd=HERE, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError((rank, p.stdout, p.stderr))
    return json.loads((HERE/f'mean_response_transport_pilot_judge_r{rank}_v1.json').read_text())
records = []
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(run, r) for r in (37, 39, 42, 48)]
    for f in concurrent.futures.as_completed(futures):
        r = f.result()
        records.append(r)
        print(json.dumps(dict(rank=r['rank'], final_mse=r['final_mse'])), flush=True)
means = {k: sum(r['final_mse'][k] for r in records)/4 for k in ('bank', 'control', 'response')}
rc, rb = means['response']/means['control'], means['response']/means['bank']
wins = sum(r['response_bank_ratio'] < 1 for r in records)
result = dict(records=sorted(records, key=lambda r: r['rank']), pooled_final_mse=means,
              new_control_ratio=rc, new_bank_ratio=rb, bank_wins=wins,
              matched_pilot_gate=rc <= .90 and rb <= .92 and wins >= 3,
              wall_s=time.time()-started,
              qualification='Fixed20/4 two-stage pilot with frozen separate P0 response, same archived matched control training configuration. Final checkpoints, actual scalar-final closed loops only. No sweep,200-fit,14-panel or adjusted/official score.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2), flush=True)
