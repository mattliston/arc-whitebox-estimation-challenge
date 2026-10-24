"""Fixed39-feature fit; launch only when the previous MPS job has finished."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
G22=HERE.parent/"public_g22_operator_features_v1"
TAG="ASTRA_COMPACT_TENSORFEAT_V1_C64_E40"
payload=HERE/"payload_compact_tensorfeat_v1.npz"
receipt=HERE/"compact_tensorfeat_training_v1.json"
if payload.exists() or receipt.exists() or (G22/f"GNN_{TAG}_MODEL.pt").exists():
    raise FileExistsError("Existing39-feature training artifacts preserved")
assert json.loads((HERE/"rows_compact_tensorfeat_v1_receipt.json").read_text())["host_all_byte_identical"]
for key in list(os.environ):
    if key.startswith("GNN_"):
        del os.environ[key]
os.environ.update(GNN_FIRST="1",GNN_HIDDEN="128",GNN_CM="16",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
trainer=G22/"train_gnn_oracle.py"
started=time.time()
subprocess.run([sys.executable,"-u",str(trainer),str(HERE/"rows_compact_tensorfeat_v1"),TAG,"40","64"],cwd=HERE,check=True)
env=dict(os.environ,EXPORT_FIRST="1",EXPORT_VHEAD="0")
subprocess.run([sys.executable,str(G22/"export_gnn_payload.py"),str(G22/f"GNN_{TAG}_MODEL.pt"),str(payload)],env=env,cwd=HERE,check=True)
receipt.write_text(json.dumps(dict(tag=TAG,wall_s=time.time()-started,trainer_sha256=hashlib.sha256(trainer.read_bytes()).hexdigest(),payload=str(payload),features=39),indent=2)+"\n")
