"""One bounded three-worker row generation and single-MPS training round."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
G22 = HERE.parent/"public_g22_operator_features_v1"
PY = "/Users/matt/arc-white-box-estimation/.venv/bin/python"
AP = "/Users/matt/anaconda3/bin/python"
ROWS = HERE/"rows_radial_covmetric_v1"
TAG = "ASTRA_RADIALCOV_V1_C64_E40"
receipt = HERE/"chain_radial_training_v1.json"
if receipt.exists() or (G22/f"GNN_{TAG}_MODEL.pt").exists():
    raise FileExistsError("Existing training artifact preserved")
frozen_names = ("k3_radial_numpy.py","radial_features.py","generate_radial_rows.py")
hashes = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in frozen_names}
started = time.time()
def row(rank):
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    run = subprocess.run([PY,"-u",str(HERE/"generate_radial_rows.py"),"--rank",str(rank),"--rows",str(ROWS),"--cov32"],
                         cwd=HERE,capture_output=True,text=True)
    if run.returncode:
        raise RuntimeError(f"Row {rank} failed: {run.stdout}\n{run.stderr}")
    data = json.loads((ROWS/f"row_{rank:04d}"/"result.json").read_text())
    return rank,data["one_step_mse"][-1]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(row,rank) for rank in range(35,235)]
    for count,future in enumerate(concurrent.futures.as_completed(futures),1):
        rank,mse = future.result()
        if count%10==0:
            print(json.dumps(dict(rows_completed=count,last_rank=rank,last_one_step_mse=mse,wall_s=time.time()-started)),flush=True)
rows = [json.loads((ROWS/f"row_{rank:04d}"/"result.json").read_text()) for rank in range(35,235)]
assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
row_seconds = time.time()-started
print("RADIAL_ROWS_COMPLETE: 200 honest rows; launching one MPS training job",flush=True)
env = {k:v for k,v in os.environ.items() if not k.startswith("GNN_")}
env.update(GNN_FIRST="1",GNN_HIDDEN="128",GNN_CM="16",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
run = subprocess.run([AP,"-u",str(G22/"train_gnn_oracle.py"),str(ROWS),TAG,"40","64"],cwd=HERE,env=env)
if run.returncode:
    raise RuntimeError(f"Training failed with code {run.returncode}")
payload = HERE/"payload_radialcov_v1.npz"
if payload.exists():
    raise FileExistsError(payload)
env.update(EXPORT_FIRST="1",EXPORT_VHEAD="0")
subprocess.run([AP,str(G22/"export_gnn_payload.py"),str(G22/f"GNN_{TAG}_MODEL.pt"),str(payload)],env=env,cwd=HERE,check=True)
receipt.write_text(json.dumps(dict(source_hashes=hashes,rows=str(ROWS),tag=TAG,row_seconds=row_seconds,
                                   total_seconds=time.time()-started,training_result=str(G22/f"GNN_{TAG}_RESULT.json"),payload=str(payload)),indent=2)+"\n")
print("RADIAL_TRAINING_ROUND_COMPLETE",flush=True)
