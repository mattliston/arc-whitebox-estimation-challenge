"""Fresh-process, fixed-seed native GOP2 price/implementation judge. Root grant only."""
import hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget

O = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader
candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve()
science_protocol_path=Path(os.environ['GOP2_SCIENCE_PROTOCOL_PATH']).resolve()
science_protocol_sha256=os.environ['GOP2_SCIENCE_PROTOCOL_SHA256']
assert hashlib.sha256(science_protocol_path.read_bytes()).hexdigest()==science_protocol_sha256
science_protocol=json.loads(science_protocol_path.read_text())
assert science_protocol['status']=='REGISTERED' and science_protocol['purpose']=='rtp_qr_e406_pilot8_fresh16_oos8'
assert science_protocol['source_pins'][str(Path(__file__).resolve())]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
phase_seeds=science_protocol['phase_seeds']
assert set(phase_seeds)=={'pilot','remaining'} and len(phase_seeds['pilot'])==8 and len(phase_seeds['remaining'])==16
allowed=phase_seeds['pilot']+phase_seeds['remaining'];assert len(set(allowed))==24
assert all(type(v)is int and 7100000<=v<7200000 for v in allowed)and seed in phase_seeds['pilot']
assert not out.exists() and not out.with_name(out.stem+'_pred.npz').exists()
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
         'VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
assert all(os.environ.get(k)=='1' for k in THREADS)
assert os.environ['PB_ALL']=='1'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==os.environ['CANDIDATE_SHA256']
mode=int(os.environ['GOP2_MODE']);assert mode in (-1,1)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
mod=types.ModuleType('candidate_cert');mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate),'exec',dont_inherit=True),mod.__dict__)
if mode==-1:assert '_GOP2_ON'not in mod.__dict__
else:assert mod._GOP2_ON is True

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
load_start=time.monotonic();w=loader.weights(seed)
truth=loader.truth(seed,check_weights=False)['mc_post']
before=hashlib.sha256(w.tobytes()).hexdigest()
mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=f'fresh-{seed}')
load_wall=time.monotonic()-load_start;ops[0]=0;start=time.monotonic()
try:
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
finally:
    _budget.BudgetContext._charge_op=original
assert y.shape==(16,1024) and np.isfinite(y).all()
after=hashlib.sha256(w.tobytes()).hexdigest();assert before==after
mse=((y-truth)**2).mean(axis=1)
row=dict(seed=seed,weights_before_sha256=before,weights_after_sha256=after,
         load_wall_s=load_wall,predict_wall_unrounded_s=time.monotonic()-start,
         flops=int(ctx.summary_dict()['flops_used']),ops=ops[0],
         mse_final=float(mse[-1]),mse_layers=[float(x) for x in mse],
         pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16],
         stats={n:json.loads(json.dumps(mod.__dict__.get(n),default=str))
                for n in os.environ['STATS'].split(',')},
         gop2_cleanup=dict(active_clear=mod.__dict__.get('_GOP2_ACTIVE',[False]) == [False], post_cov32_absent='post_cov32' not in mod._E2_LAYER,
                          hcf_ctx_clear=mod._HCF_STATE['ctx'] is None,
                          hcf_rows_clear=mod._HCF_STATE['rows15'] is None,
                          k4l_final_clear=mod._K4L_FINAL[0] is None))
np.savez(str(out).replace('.json','_pred.npz'),**{str(seed):y.astype(np.float32)})
with out.open('x') as f:
    json.dump(dict(candidate=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                   seeds=[seed],rows=[row],setup=setup,gop2_mode=mode,
                   science_protocol_path=str(science_protocol_path),science_protocol_sha256=science_protocol_sha256,
                   judge_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   loader_sha256=hashlib.sha256(Path(loader.__file__).read_bytes()).hexdigest(),
                   environment={k:os.environ[k] for k in THREADS+('PB_ALL','CANDIDATE_SHA256','GOP2_MODE')},
                   native_bill_comparable=True,production_wall_valid=False,
                   prediction_dtype='float32',python_version=sys.version),f)
print(json.dumps(dict(done=True,seed=seed,mode=mode,flops=row['flops'])),flush=True)
