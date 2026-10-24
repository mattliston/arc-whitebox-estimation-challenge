"""Three CPU row workers only; the final MPS fit is a separate command."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
PY="/Users/matt/arc-white-box-estimation/.venv/bin/python"
receipt=HERE/"fullpair_final_rows_v1.json"
if receipt.exists():
    raise FileExistsError(receipt)
names=("radial_compact_feed_fullpair_v1.py","radial_compact_radialfeed_v1.py","radial_mehler.py","radial_binary_compact.py","radial_features.py","radial_gnn_numpy.py","compact_final_head_v1.py","generate_fullpair_final_rows_v1.py","payload_compact_feed_fullpair_v1.npz")
hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
started=time.time()


def row(rank):
    assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    run=subprocess.run([PY,"-u",str(HERE/"generate_fullpair_final_rows_v1.py"),"--rank",str(rank)],cwd=HERE,capture_output=True,text=True)
    if run.returncode:
        raise RuntimeError(f"rank{rank}: {run.stdout}\n{run.stderr}")
    return json.loads((HERE/"rows_fullpair_final_head_v1"/f"row_{rank:04d}"/"result.json").read_text())


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(row,r) for r in range(35,235)]
    for count,future in enumerate(concurrent.futures.as_completed(futures),1):
        rec=future.result()
        if count%10==0:
            print(json.dumps(dict(rows_completed=count,last_rank=rec["rank"],old_mse=rec["old_final_mse"],wall_s=time.time()-started)),flush=True)
receipt.write_text(json.dumps(dict(source_hashes=hashes,rows=200,wall_s=time.time()-started),indent=2)+"\n")
print("FULLPAIR_FINAL_ROWS_COMPLETE; MPS fit NOT started",flush=True)
