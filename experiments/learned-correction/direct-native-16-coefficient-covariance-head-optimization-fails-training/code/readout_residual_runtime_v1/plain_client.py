"""Ordinary NumPy-free client; no profiler, RPC wrapper, or timer changes."""
import hashlib,importlib.util,json,sys,time,types
from pathlib import Path
import flopscope as flops
import flopscope.numpy as fnp
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();ranks=list(map(int,sys.argv[3].split(',')))
out.mkdir(exist_ok=False)
assert importlib.util.find_spec('numpy') is None
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('ordinary_candidate',candidate);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
manifest=json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
records=[]
for index,rank in enumerate(ranks):
    path=STORE/Path(manifest[rank]['path']).parent/'weights.npy'
    with flops.BudgetContext(flop_budget=2**41,quiet=True,namespace=f'ingress-{index}'):
        blob=fnp.load(str(path));weights=[blob[l] for l in range(16)]
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=weights,seed=0,name=f'rank{rank}')
    started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True,namespace=f'predict-{index}') as ctx:
        if index==0:spec.loader.exec_module(mod);estimator=mod.Estimator()
        prediction=estimator.predict(mlp,2**41)
    wall=time.monotonic()-started;summary=ctx.summary_dict()
    output=out/f'call{index:02d}_rank{rank:04d}.npy'
    with flops.BudgetContext(flop_budget=2**41,quiet=True,namespace=f'save-{index}'):
        fnp.save(str(output),prediction)
    record=dict(index=index,rank=rank,output=str(output),wall_s=wall,summary=summary,
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),includes_initialization=index==0)
    records.append(record)
    with (out/f'call{index:02d}.json').open('x') as f:json.dump(record,f,indent=2)
    print(dict(index=index,rank=rank,flops=summary['flops_used'],wall_s=wall,
        residual=summary['residual_wall_time_s']),flush=True)
    del prediction,mlp,weights,blob
with (out/'completed.json').open('x') as f:json.dump(dict(completed=True,candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    records=records,truth_accessed=False,numpy_available=False,profiling=False,accounting_modified=False),f,indent=2)
