"""Three low-priority CPU row workers, then one MPS final-head fit."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
PY = "/Users/matt/arc-white-box-estimation/.venv/bin/python"
AP = "/Users/matt/anaconda3/bin/python"
receipt = HERE / "chain_compact_final_head_v1.json"
if receipt.exists() or (HERE / "payload_compact_final_head_v1.npz").exists():
    raise FileExistsError("Existing final-only round preserved")
names = ("radial_binary_compact.py", "radial_features.py", "radial_gnn_numpy.py", "compact_final_head_v1.py", "generate_compact_final_rows_v1.py", "payload_binary_compact_g01_v1.npz")
hashes = {n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
started = time.time()


def row(rank):
    assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest() == digest for n,digest in hashes.items())
    run = subprocess.run([PY,"-u",str(HERE/"generate_compact_final_rows_v1.py"),"--rank",str(rank)], capture_output=True, text=True, cwd=HERE)
    if run.returncode:
        raise RuntimeError(f"rank{rank}: {run.stdout}\n{run.stderr}")
    return json.loads((HERE/"rows_compact_final_head_v1"/f"row_{rank:04d}"/"result.json").read_text())


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(row, r) for r in range(35,235)]
    for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
        rec = future.result()
        if count%10 == 0:
            print(json.dumps(dict(rows_completed=count, last_rank=rec["rank"], old_mse=rec["old_final_mse"], wall_s=time.time()-started)), flush=True)
row_seconds = time.time()-started
env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
subprocess.run([AP,"-u",str(HERE/"train_compact_final_head_v1.py")], env=env, cwd=HERE, check=True)
receipt.write_text(json.dumps(dict(source_hashes=hashes, row_seconds=row_seconds, wall_s=time.time()-started), indent=2)+"\n")
print("COMPACT_FINAL_HEAD_ROUND_COMPLETE", flush=True)
