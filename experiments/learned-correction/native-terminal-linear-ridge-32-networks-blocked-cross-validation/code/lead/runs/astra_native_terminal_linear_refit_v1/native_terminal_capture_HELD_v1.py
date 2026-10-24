"""Fresh-process, fixed-seed native GOP2 price/implementation judge. Root grant only."""
import hashlib, json, os, sys, time, types, importlib.util
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget

O = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader
candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve()
assert seed in range(32)
assert not out.exists() and not out.with_name(out.stem+'_pred.npz').exists()
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
         'VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
assert all(os.environ.get(k)=='1' for k in THREADS)
assert os.environ['PB_ALL']=='1'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==os.environ['CANDIDATE_SHA256']
mode=int(os.environ['GOP2_MODE']);assert mode == 1
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
mod=types.ModuleType('candidate_cert');mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate),'exec',dont_inherit=True),mod.__dict__)
assert mod._GOP2_ON is True

mod._PB_WALL_S=1e9
est=mod.Estimator();start=time.monotonic()
est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
setup=dict(wall_s=time.monotonic()-start,pb_log=json.loads(json.dumps(mod.__dict__.get('_PB_LOG'),default=str)),
           pb_single_sha256=hashlib.sha256(repr(mod._PB_SINGLE).encode()).hexdigest(),
           pb_single_count=len(mod._PB_SINGLE),pb_wall_s=mod._PB_WALL_S,
           cache_counts={name:len(mod.__dict__[name]) for name in mod._PB_CACHES
                         if name in mod.__dict__ and hasattr(mod.__dict__[name],'__len__')})
ops=[0];original=_budget.BudgetContext._charge_op
def counted(self,*a,**kw):
    ops[0]+=1
    return original(self,*a,**kw)
_budget.BudgetContext._charge_op=counted
load_start=time.monotonic()
config=json.loads((Path(__file__).resolve().parent/'CAPTURE_PROTOCOL_REGISTERED_v1.json').read_text())
config=next(row for row in config['network_metadata'] if row['rank']==seed)
assert hashlib.sha256(Path(config['weights_path']).read_bytes()).hexdigest()==config['weights_sha256']
w=np.load(config['weights_path'],allow_pickle=False)
assert w.shape==(16,1024,1024) and w.dtype==np.float32
before=hashlib.sha256(w.tobytes()).hexdigest();assert before==config['weights_raw_sha256']
mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=f'fresh-{seed}')
load_wall=time.monotonic()-load_start;ops[0]=0;start=time.monotonic()
if seed==16:
    helper=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_read_column_oracle_v1/observer_stream_HELD_v1.py')
    spec=importlib.util.spec_from_file_location('_read_observer',helper);obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)
    mod._RCO_OBSERVER=obs.make_observer(out.parent/'train16_read_observer_v1')
try:
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
finally:
    _budget.BudgetContext._charge_op=original
    mod._RCO_OBSERVER=None
assert mod._RCO_CONTEXT==[None]
assert y.shape==(16,1024) and np.isfinite(y).all()
after=hashlib.sha256(w.tobytes()).hexdigest();assert before==after
predict_wall=time.monotonic()-start
assert mod._TLR_ACTIVE is False and mod._RTP_ON is True
assert mod._RTT_ACTIVE is False and mod._RTT_CONTEXT[0] is None
assert mod._RTT_STATS['retired'] is True and mod._GOP2_STATS['retired'] is True
assert mod._GOP2_ACTIVE == [False]
cap={key:np.array(value,copy=True) for key,value in mod._TLR_CAPTURE.items()}
assert cap['features30'].shape==(1024,30) and cap['state64'].shape==(1024,64)
assert cap['sigma'].shape==cap['host'].shape==cap['delta'].shape==(1024,)
assert all(np.isfinite(a).all() for a in cap.values()) and np.all(cap['sigma']>0)
raw=cap['host'].astype(np.float64)+cap['delta'].astype(np.float64)
assert np.maximum(raw,0).astype(np.float32).tobytes()==y[15].astype(np.float32).tobytes()
cap['raw_final_float64']=raw
cap['predictions_float32']=y.astype(np.float32)
# Target read is TRAIN-only and outside the charged native prediction context.
assert hashlib.sha256(Path(config['truth_path']).read_bytes()).hexdigest()==config['truth_sha256']
with np.load(config['truth_path'],allow_pickle=False) as truth:
    assert int(truth['mlp_id'])==config['mlp_id']
    cap['target_final_float64']=truth['mc_post'][15].astype(np.float64)
cp=out.with_name(out.stem+'_capture.npz');assert not cp.exists();np.savez(cp,**cap)
mod._TLR_CAPTURE.clear()
row=dict(seed=seed,weights_before_sha256=before,weights_after_sha256=after,
         load_wall_s=load_wall,predict_wall_unrounded_s=time.monotonic()-start,
         flops=int(ctx.summary_dict()['flops_used']),ops=ops[0],
         pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16],
         stats={n:json.loads(json.dumps(mod.__dict__.get(n),default=str))
                for n in os.environ['STATS'].split(',')},
         gop2_cleanup=dict(active_clear=mod._GOP2_ACTIVE == [False], post_cov32_absent='post_cov32' not in mod._E2_LAYER,
                          hcf_ctx_clear=mod._HCF_STATE['ctx'] is None,
                          hcf_rows_clear=mod._HCF_STATE['rows15'] is None,
                          k4l_final_clear=mod._K4L_FINAL[0] is None))
np.savez(str(out).replace('.json','_pred.npz'),**{str(seed):y.astype(np.float32)})
with out.open('x') as f:
    json.dump(dict(capture_complete=True,capture_path=str(cp),capture_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
                   layers=[15],terminal94_complete=True,all16_finite=True,all16_parity_measured=False,source_only_path_equivalence=True,
                   weights_before_sha256=before,weights_after_sha256=after,flops=row['flops'],predict_wall_s=predict_wall,
                   candidate=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                   seeds=[seed],rows=[row],setup=setup,gop2_mode=mode,
                   judge_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   loader_sha256=hashlib.sha256(Path(loader.__file__).read_bytes()).hexdigest(),
                   environment={k:os.environ[k] for k in THREADS+('PB_ALL','CANDIDATE_SHA256','GOP2_MODE')},
                   native_bill_comparable=True,production_wall_valid=False,observational_FNP_calls=0,observational_flops=0,
                   prediction_dtype='float32',python_version=sys.version),f)
print(json.dumps(dict(done=True,seed=seed,mode=mode,flops=row['flops'])),flush=True)
