"""Versioned independent-stream generation using preserved local MC code."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,choices=(35,100),required=True);args=ap.parse_args()
if args.rank==100:
    first=json.loads((HERE/'covariance_innovation_replica_r35_v1.json').read_text())
    assert first['run_rank100_gate']
folder=HERE/'covariance_replica_truth_v1';folder.mkdir(exist_ok=True)
out=folder/f'rank_{args.rank:04d}_8M_v1.npz';receipt=folder/f'rank_{args.rank:04d}_8M_v1.json'
assert not out.exists() and not receipt.exists()
source=HERE.parent/'public_g19_attribution_v1/mc_truth_mps_indep.py'
seed_xor=0x4153545241434F56
spec=importlib.util.spec_from_file_location('covariance_replica_primary_mc_v1',source)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.SEED_XOR=seed_xor
torch.set_num_threads(4)
assert torch.backends.mps.is_available()
sys.argv=[str(source),str(args.rank),'8',str(out),'cov']
t0=time.time();print(json.dumps(dict(rank=args.rank,start=True,seed_xor=hex(seed_xor),batch=mod.BATCH)),flush=True)
mod.main()
record=dict(rank=args.rank,samples=8000000,seed_xor=hex(seed_xor),rng='PCG64DXSM CPU float32 standard_normal',
            source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),batch=mod.BATCH,
            wall_s=time.time()-t0,torch_version=torch.__version__,qualification='Independent sample stream, same numerical MC implementation; not independent numerical precision or model bias.')
receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
