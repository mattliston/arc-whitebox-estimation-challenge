"""Finite24-row gate; three nice10 workers, each limited to two BLAS threads."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import time

here = Path(__file__).resolve().parent
out = here/'conditional_quadratic_tensor_probe_rows_v1.json'
assert not out.exists()
started = time.time()


def one(rank):
    cmd = ['nice', '-n', '10', '/Users/matt/arc-white-box-estimation/.venv/bin/python', '-u',
           str(here/'generate_conditional_quadratic_tensor_probe_v1.py'), '--rank', str(rank)]
    run = subprocess.run(cmd, cwd=here, capture_output=True, text=True)
    print(run.stdout, end='', flush=True)
    assert run.returncode == 0, (rank, run.stderr)
    r = json.loads((here/'rows_conditional_quadratic_tensor_probe_v1'/f'row_{rank:04d}'/'result.json').read_text())
    assert r['parent_byte_exact']
    return r


with ThreadPoolExecutor(max_workers=3) as pool:
    rows = list(pool.map(one, range(35, 59)))
out.write_text(json.dumps(dict(n=24, records=rows, wall_s=time.time()-started), indent=2)+'\n')
print('CONDITIONAL_QUADRATIC_TENSOR_ROWS_COMPLETE', flush=True)
