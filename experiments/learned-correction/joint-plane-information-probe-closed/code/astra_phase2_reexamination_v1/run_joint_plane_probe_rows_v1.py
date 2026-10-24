"""Two nice-inherited local workers; no training, submission, or overwrites."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
target=HERE/"joint_plane_probe_rows_v1.json"
assert not target.exists()
def row(rank):
    result=subprocess.run([sys.executable,"-u",str(HERE/"generate_joint_plane_probe_v1.py"),"--rank",str(rank)],cwd=HERE,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return json.loads((HERE/"rows_joint_plane_probe_v1"/f"row_{rank:04d}"/"result.json").read_text())
t0=time.time();rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(row,rank) for rank in range(35,59)]
    for future in concurrent.futures.as_completed(futures):
        rows.append(future.result())
        print(json.dumps(dict(completed=len(rows),wall_s=time.time()-t0)),flush=True)
target.write_text(json.dumps(dict(rows=rows,wall_s=time.time()-t0),indent=2)+"\n")
