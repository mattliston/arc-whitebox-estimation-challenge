from pathlib import Path
import hashlib,json,os,re,resource,sys,threading,time,types,subprocess
import numpy as np
import psutil
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
O=Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1")
S=Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py")
START=time.monotonic();DONE=threading.Event();PEAK=[0]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
 with (D/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def free():
 out=subprocess.check_output(['/usr/bin/memory_pressure'],text=True,timeout=10)
 return int(re.search(r'System-wide memory free percentage: (\d+)%',out).group(1))
assert all(os.environ.get(x)=='1' for x in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'))
assert os.getpriority(os.PRIO_PROCESS,0)==10
assert free()>=50
assert sha(S)=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
tariff=json.loads((D/'CENTERING_TARIFF_v1.json').read_text())
assert tariff['fifteen_full_layers_upper_percent']<.2
ret=O/'lead/runs/astra_join_residual_selector_science_v1/PILOT_RETIREMENT_astra_v1.json'
assert sha(ret).startswith('67bc4eab')
write('PROCESS_v1.json',dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgid(0),nice=os.getpriority(os.PRIO_PROCESS,0),source_sha256=sha(__file__),candidate_sha256=sha(S),retirement_path=str(ret),retirement_sha256=sha(ret)))
def monitor():
 p=psutil.Process()
 while not DONE.wait(.25):
  rss=p.memory_info().rss;PEAK[0]=max(PEAK[0],rss)
  if rss>7_500_000_000 or time.monotonic()-START>120 or psutil.virtual_memory().available/psutil.virtual_memory().total<.5:
   write('GUARD_STOP_v1.json',dict(rss=rss,seconds=time.monotonic()-START))
   os._exit(92)
threading.Thread(target=monitor,daemon=True).start()
# One fixed historical TRAIN16, entering layer14. No current-state/full-model claim.
cp=O/'physicist/t128_cdump/t128_C_16_L14.npz'
with np.load(cp,allow_pickle=False)as z:C64=z['C'];mu=z['mu']
raw_asym=float(np.max(np.abs(C64-C64.T)))
C=((C64+C64.T)*.5).astype('f4')
protocol=json.loads((O/'lead/runs/astra_current_variance_capture_v1/CAPTURE_PROTOCOL_HELD_v1.json').read_text())
wp=Path(protocol['weights_path']);weights=np.load(wp,allow_pickle=False)
assert hashlib.sha256(weights.tobytes()).hexdigest()==protocol['weights_raw_sha256']
W=weights[14].astype('f4')
assert C.shape==W.shape==(1024,1024)
# Exact current dispatch source, no setup/predict and no altered kernels.
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
M=types.ModuleType('centering_current');M.__file__=str(S);sys.modules[M.__name__]=M
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 exec(compile(S.read_text(),str(S),'exec'),M.__dict__)
 arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32)
 M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None]
 M._bind_push6(arena)
 Cfp=fnp.asarray(C);Wfp=fnp.asarray(W);Wt=Wfp.T
M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
assert M._WINOGRAD and M._AB_ON
assert M._sym_upper.__name__=='_psdb_sym_upper'
M._TR_LAST[0]=None
with fs.BudgetContext(flop_budget=2**60,quiet=True)as b0:
 first=M._mm(Cfp,Wfp);original=M._sym_upper(Wt,first)
original=np.asarray(original).copy()
M._TR_LAST[0]=None
with fs.BudgetContext(flop_budget=2**60,quiet=True)as b1:
 r=fnp.mean(Cfp,axis=1);c=fnp.mean(r)
 C0=((Cfp-r[:,None])-r[None,:])+c
 first0=M._mm(C0,Wfp);centered_core=M._sym_upper(Wt,first0)
 u=fnp.sum(Wfp,axis=0);v=r@Wfp;t=v-(c*.5)*u
 R=t[:,None]*u[None,:]
 centered=(centered_core+R)+R.T
centered=np.asarray(centered).copy()
assert np.array_equal(np.asarray(Cfp),C)and np.array_equal(np.asarray(Wfp),W)
# Float64 reference uses identical FP32 inputs; not original unrounded C64.
Cref=C.astype('d');Wref=W.astype('d');reference=Wref.T@(Cref@Wref)
r64=Cref.mean(axis=1);c64=r64.mean();C064=Cref-r64[:,None]-r64[None,:]+c64
u64=Wref.sum(axis=0);v64=r64@Wref;t64=v64-.5*c64*u64
refcenter=Wref.T@(C064@Wref)+t64[:,None]*u64[None,:]+u64[:,None]*t64[None,:]
identity=float(np.max(np.abs(refcenter-reference)))
mean=mu@Wref;sigma=np.sqrt(np.maximum(np.diag(reference),1e-30));sensitivity=np.exp(-.5*(mean/sigma)**2)/(2*np.sqrt(2*np.pi)*sigma)
def metrics(x):
 e=x.astype('d')-reference;diag=np.diag(e);off=e.copy();np.fill_diagonal(off,0)
 return dict(relative_fro_error=float(np.linalg.norm(e)/np.linalg.norm(reference)),rms_error=float(np.sqrt(np.mean(e**2))),max_abs_error=float(np.max(np.abs(e))),diagonal_rms_error=float(np.sqrt(np.mean(diag**2))),offdiagonal_rms_error=float(np.sqrt(np.mean(off**2))),local_gaussian_mean_shift_rms=float(np.sqrt(np.mean((diag*sensitivity)**2))))
a=metrics(original);b=metrics(centered)
PEAK[0]=max(PEAK[0],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
result=dict(status='ONE_FIXED_HISTORICAL_NUMERICAL_CHECK_COMPLETE',network=16,next_layer=14,shape=[1024,1024],covariance_path=str(cp),covariance_sha256=sha(cp),weights_path=str(wp),weights_raw_sha256=protocol['weights_raw_sha256'],current_kernel_source_sha256=sha(S),original=a,centered=b,centered_over_original_squared_Fro_error=(b['relative_fro_error']/a['relative_fro_error'])**2,centered_over_original_diagonal_rms=b['diagonal_rms_error']/a['diagonal_rms_error'],F64_identity_max_error=identity,source_covariance_asymmetry_max=raw_asym,original_flops=int(b0.flops_used),centered_flops=int(b1.flops_used),original_ops=len(b0._op_log),centered_ops=len(b1._op_log),inputs_unchanged=True,seconds=time.monotonic()-START,peak_rss_bytes=PEAK[0],new_model_calls=0,new_truths=0,limits=['Historical S47e full1024 covariance, no saved dead-row roster; not current1c913 state.','Current source _mm and _sym_upper numeric dispatch; only algebraic centering differs.','Input symmetrized then castF32; F64 reference uses those sameF32inputs.','No finalMSE or adjusted-score claim; local mean displacement is a sensitivity proxy.','Warm extra tariff is separately measured; cold kernel bindings differ between two calls.'])
write('CENTERING_ROUNDOFF_RESULT_v1.json',result);DONE.set()
print(json.dumps(result));print('receipt_sha256',sha(D/'CENTERING_ROUNDOFF_RESULT_v1.json'))

