"""New shape/billing metadata, unchanged frozen bank and predictions.

Read-only numerical diagnostic. No labels or submission. Wrapper overhead
is explicitly NOT a persistent grader-runtime qualification.
"""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE=Path(__file__).resolve().parent;OUT=HERE/'bank_matmul_census_v1.json';assert not OUT.exists()
SOURCE=HERE/'candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py'
old=json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_native_v1.json').read_text())
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==old['source_sha256']
STORE=HERE.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man=json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
spec=importlib.util.spec_from_file_location('bank_shape_census_v1',SOURCE)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
flops.configure(symmetry_warnings=False)
original=mod._mm
records=[];started=time.time()
for call,rank in enumerate((35,100)):
    summary=defaultdict(lambda:dict(calls=0,flops=0))
    def wrapped(a,b):
        m,k=int(a.shape[0]),int(a.shape[1]);n=int(b.shape[1])
        leaf=n<=mod._PACKED_TILE and k<=mod._PACKED_TILE
        before=ctx.flops_used if leaf else None
        result=original(a,b)
        if leaf:
            key=f'{m},{k},{n};{a.dtype};{b.dtype}'
            summary[key]['calls']+=1;summary[key]['flops']+=int(ctx.flops_used-before)
        return result
    mod._mm=wrapped
    rp=STORE/man[rank]['path'];row=json.loads(rp.read_text())
    weights=np.load(rp.parent/'weights.npy',mmap_mode='r')
    mlp=MLP(width=1024,depth=16,weights=list(weights),seed=int(derive_estimator_seed(int(row['mlp_seed']))),name=f'bank-shapes-{rank}')
    t0=time.time()
    with threadpool_limits(limits=4):
        with flops.BudgetContext(flop_budget=2**41,wall_time_limit_s=120,quiet=True) as ctx:
            result=mod.Estimator().predict(mlp,2**41)
    output=np.asarray(result,np.float64)
    digest=hashlib.sha256(output.tobytes()).hexdigest()
    expected=next(x['output_sha256'] for x in old['trials'] if x['rank']==rank)
    assert digest==expected,('frozen_prediction_hash',rank,digest,expected)
    records.append(dict(rank=rank,flops=int(ctx.flops_used),shapes=dict(summary),
                        output_sha256=digest,old_prediction_byte_exact=True,wall_s=time.time()-t0))
    print(json.dumps({k:v for k,v in records[-1].items() if k!='shapes'}),flush=True)
result=dict(records=records,wall_s=time.time()-started,source_sha256=old['source_sha256'],
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='Fresh metadata from frozen bank; unchanged output hashes; no truth accessed, raw score or runtime qualification.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
