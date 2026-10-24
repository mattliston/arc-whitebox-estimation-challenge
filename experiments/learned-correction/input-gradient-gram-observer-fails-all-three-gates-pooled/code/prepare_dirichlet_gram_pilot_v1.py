"""At most two inherited-nice input-only workers; seal all24 fixed rows."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / 'dirichlet_gram_rows_pilot_v1.json'
PROGRESS = HERE / 'dirichlet_gram_row_preparation_progress_v1'
assert not OUT.exists() and not PROGRESS.exists()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for smoke_rank in (35, 100):
    folder = HERE / 'rows_dirichlet_gram_probe_v1' / f'row_{smoke_rank:04d}'
    rec = json.loads((folder / 'result.json').read_text())
    assert rec['passed'] and rec['deterministic_replay'] and rec['inputs_unchanged']
    assert rec['data_sha256'] == digest(folder / 'data.npz')
source_paths = [HERE / 'dirichlet_gram_observer_v1.py', HERE / 'generate_dirichlet_gram_row_v1.py',
                HERE / 'DIRICHLET_GRAM_INFORMATION_DESIGN_V1.md',
                HERE / 'DIRICHLET_GRAM_ROOT_ACCEPTANCE_V1.md', Path(__file__).resolve()]
source_hashes = {p.name: digest(p) for p in source_paths}
PROGRESS.mkdir()
started = time.time()


def prepare(rank):
    result = subprocess.run([sys.executable, str(HERE / 'generate_dirichlet_gram_row_v1.py'),
                             '--rank', str(rank)], cwd=HERE, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    run = dict(rank=rank, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    (PROGRESS / f'row_{rank:04d}.json').write_text(json.dumps(run, indent=2) + '\n')
    assert result.returncode == 0, run
    folder = HERE / 'rows_dirichlet_gram_probe_v1' / f'row_{rank:04d}'
    rec = json.loads((folder / 'result.json').read_text())
    assert rec['passed'] and rec['target_access'] is False and rec['inputs_unchanged']
    assert all(rec['byte_parity'].values())
    assert rec['data_sha256'] == digest(folder / 'data.npz')
    answer = dict(rank=rank, passed=True, data_sha256=rec['data_sha256'],
                  record_sha256=digest(folder / 'result.json'), wall_s=rec['wall_s'],
                  peak_rss_bytes=rec['peak_rss_bytes'])
    print(json.dumps(answer), flush=True)
    return answer


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    records = list(executor.map(prepare, range(35, 59)))
assert source_hashes == {p.name: digest(p) for p in source_paths}
receipt = dict(passed=len(records) == 24 and all(r['passed'] for r in records),
               target_access=False, rows=records, ranks=list(range(35, 59)),
               source_sha256=source_hashes, source_unchanged=True,
               max_workers=2, wall_s=time.time() - started,
               qualification='All24 input-only rows sealed; no target fit or score yet.')
OUT.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items() if k not in ('rows', 'source_sha256')}, indent=2), flush=True)
