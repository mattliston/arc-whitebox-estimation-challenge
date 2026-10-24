"""Five real remote contexts in one NumPy-free participant process."""
import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import flopscope as flops
import flopscope.numpy as fnp

HERE=Path(__file__).resolve().parent
STORE=HERE.parent/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--tag",required=True)
args=ap.parse_args()
outfile=HERE/f"{args.tag}.json"
assert not outfile.exists()
assert importlib.util.find_spec("numpy") is None
stub=types.ModuleType("whestbench")
class BaseEstimator: pass
stub.BaseEstimator=BaseEstimator;sys.modules["whestbench"]=stub
source=HERE/args.candidate
spec=importlib.util.spec_from_file_location("astra_remote_series_candidate",source)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
man=json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]

@dataclass
class MLP:
    width:int
    depth:int
    weights:list
    seed:int=0
    name:str="phase2-remote-series"

estimator=mod.Estimator()

def run_one(index,rank):
    if rank<1000:
        fixture=STORE/Path(man[rank]["path"]).parent/"weights.npy"
    else:
        fixture=HERE.parent/"public_g19_attribution_v1"/f"fresh_{rank}_weights.npy"
    assert fixture.is_file()
    with flops.BudgetContext(flop_budget=2**41,quiet=True,namespace=f"{args.tag}-{index}") as ctx:
        blob=fnp.load(str(fixture));weights=[blob[l] for l in range(16)]
        out=estimator.predict(MLP(1024,16,weights),2**41)
        raw,shape,dtype=out._fetch_data()
    summary=ctx.summary_dict()
    rawfile=HERE/f"{args.tag}_{index}_r{rank}.f32"
    assert not rawfile.exists()
    rawfile.write_bytes(raw)
    return dict(index=index,rank=rank,output_shape=list(shape),output_dtype=str(dtype),
                output_sha256=hashlib.sha256(raw).hexdigest(),
                flops=int(summary["flops_used"]),util=int(summary["flops_used"])/2**41,
                timing={key:value for key,value in summary.items() if "time" in key})

records=[];started=time.time()
try:
    for index,rank in enumerate((35,100,500,3016500,35)):
        record=run_one(index,rank)
        records.append(record)
        print(json.dumps(record),flush=True)
    assert records[0]["output_sha256"]==records[-1]["output_sha256"], "Repeated rank 35 output changed"
    status="completed"
except Exception as error:
    status="failed"
    outfile.write_text(json.dumps(dict(status=status,error_type=type(error).__name__,error=str(error),records=records),indent=2)+"\n")
    raise
outfile.write_text(json.dumps(dict(status=status,candidate=args.candidate,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),records=records,
    repeated_output_byte_identical=True,wall_s=time.time()-started,
    scope="One persistent NumPy-free client/server; test only, no deployed backend modification."),indent=2)+"\n")

