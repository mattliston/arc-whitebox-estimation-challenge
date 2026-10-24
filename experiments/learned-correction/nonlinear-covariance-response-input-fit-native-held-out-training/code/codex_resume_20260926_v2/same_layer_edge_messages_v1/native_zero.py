"""Check both paid zero extensions against independently captured native S31."""
from pathlib import Path
import ctypes
import gc
import hashlib
import json
import sys
import time
import types
import numpy as np
import flopscope as flops
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((HERE/'protocol.json').read_text())
assert not (HERE/'native_zero.json').exists()
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
w=np.load(store/Path(manifest[38]['path']).parent/'weights.npy')
mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name='phase2-38')
reference=H2/'response_message_capture_train12_v1';baseline=np.load(reference/'pred_38.npy')
brow=[r for r in map(json.loads,(reference/'rows.jsonl').read_text().splitlines()) if r['rank']==38][0]
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
records=[]
for mode in ('message','control'):
    built=json.loads((HERE/f'build_{mode}_zero.json').read_text());path=Path(built['candidate']);assert sha(path)==built['sha256']
    mod=types.ModuleType('rnm_zero_'+mode);mod.__file__=str(path);sys.modules[mod.__name__]=mod
    exec(compile(path.read_text(),str(path),'exec'),mod.__dict__)
    est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
    start=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
    used=int(ctx.summary_dict()['flops_used'])
    assert np.array_equal(y,baseline),(mode,float(np.max(np.abs(y-baseline))))
    assert used>brow['flops']
    row=dict(mode=mode,rank=38,all16_bytes_exact=True,max_abs=0.,flops=used,base_flops=brow['flops'],
             bill_pct=100*(used/brow['flops']-1),wall_s=time.monotonic()-start,sha256=built['sha256'],targets_accessed=False)
    records.append(row);print(json.dumps(row),flush=True)
    del sys.modules[mod.__name__];del mod,est,y;gc.collect();relief(None,0)
with (HERE/'native_zero.json').open('x') as fp:json.dump(dict(passed=True,records=records,protocol_sha256=sha(HERE/'protocol.json'),script_sha256=sha(HERE/'native_zero.py')),fp,indent=2)
