"""One saved full-shape read; complete pair rotation, actual public tariff."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):
    os.environ[key]='1'
import gc, hashlib, json, math, resource, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp

D=Path(__file__).resolve().parent; R=D.parent; O=R.parents[1]
S=R/'astra_saved_error_mechanism_v1/run_replay_train16_v1/native_16_capture_shadow'
SOURCE=R/'opus_fix336687_v1/candidate_v61j.py'
start=time.monotonic()
z=np.load(S/'L04_read.npz')
left=z['left'].astype('f4'); lam=z['lam'].astype('f4'); cols=z['cols']
c15=int(z['c15']); right=z['right'][:c15].astype('f4'); rows=z['rows']
m,k=left.shape; n=right.shape[0]
assert m>0 and n>0
cuts=np.r_[0,np.flatnonzero(np.signbit(lam[1:])!=np.signbit(lam[:-1]))+1,len(lam)]
cohorts=[]
for i in range(0,len(cuts)-1,2):
    a,b,c=map(int,cuts[i:i+3]); assert b-a==c-b and np.array_equal(lam[a:b],-lam[b:c]); cohorts.append((a,b-a))
A=np.asarray(left*lam if m<n else left,dtype='f4'); B=np.asarray(right.T if m<n else (right*lam).T,dtype='f4')
# The one fixed balancing rule is based on operand norms, never endpoint truth.
U0=[]; U1=[]; V0=[]; V1=[]
for a,h in cohorts:
    ap=A[:,a:a+h]; am=A[:,a+h:a+2*h]; bp=B[a:a+h]; bm=B[a+h:a+2*h]
    U0.append(ap+am); U1.append(ap-am); V0.append((bp+bm)*np.float32(.5)); V1.append((bp-bm)*np.float32(.5))
norm=lambda seq: math.sqrt(sum(float(np.sum(x.astype('f8')**2)) for x in seq))
norms=[norm(v) for v in (U0,U1,V0,V1)]
exponent=int(round(.5*math.log2((norms[1]*norms[2])/(norms[0]*norms[3]))))
scale=float(2.**exponent)
del U0,U1,V0,V1; gc.collect()
prereg=dict(layer=4,saved_rank=16,shape=[m,k,n],cohorts=cohorts,scale_rule='2**round(.5*log2(||U1||*||V0||/(||U0||*||V1||)))',scale=scale,operand_norms=norms,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),actual_version=fs.__version__,models=0,comparison='Current production peel Smirnov read route, complete signed rotation; no compression/APA, both channels retained; historical saved inputs.',current_best=1.4515052706908603e-9)
(D/'PAIR_ROTATION_PREREG_v3.json').open('x').write(json.dumps(prereg,indent=2)+'\n')
print('PREREG',json.dumps(prereg),flush=True)
reference=A.astype('f8')@B.astype('f8')
stub=types.ModuleType('whestbench'); stub.BaseEstimator=type('BaseEstimator',(),{}); sys.modules['whestbench']=stub
M=types.ModuleType('pair_rotation_current'); M.__file__=str(SOURCE); sys.modules[M.__name__]=M
exec(compile(SOURCE.read_text(),str(SOURCE),'exec'),M.__dict__)
key=(m,k,n,0)
plan=M._PEEL_READ_PLANS[key]
assert (m,k,n) not in M._R7_SHAPES
M._WINOGRAD=True; M._AB_ON=True
with fs.BudgetContext(flop_budget=2**50,quiet=True):
    arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32)
    M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None]
    M._bind_push6(arena)
    M._ASTRA_SM40_ENV._SCRATCH=M._SCRATCH
    M._astra_sm40_bind_universal(M._ASTRA_SM40_ENV,arena)
    M._smx_bind()
    a=fnp.asarray(A); b=fnp.asarray(B)
del A,B,left,right,z; gc.collect()
results=[]; outputs=[]
for rotation in (False,True):
    M._BH_HOLD.clear(); M._TR_LAST[0]=None
    before=time.monotonic()
    with fs.BudgetContext(flop_budget=2**50,quiet=True) as budget:
        if rotation:
            aa=fnp.empty((m,k),dtype=fnp.float32); bb=fnp.empty((k,n),dtype=fnp.float32)
            for st,h in cohorts:
                ap=a[:,st:st+h]; am=a[:,st+h:st+2*h]; bp=b[st:st+h]; bm=b[st+h:st+2*h]
                fnp.add(ap,am,out=aa[:,st:st+h]); fnp.multiply(aa[:,st:st+h],scale,out=aa[:,st:st+h])
                fnp.subtract(ap,am,out=aa[:,st+h:st+2*h])
                fnp.add(bp,bm,out=bb[st:st+h]); fnp.multiply(bb[st:st+h],.5/scale,out=bb[st:st+h])
                fnp.subtract(bp,bm,out=bb[st+h:st+2*h]); fnp.multiply(bb[st+h:st+2*h],.5,out=bb[st+h:st+2*h])
        else: aa,bb=a,b
        transformed=np.asarray(aa).astype('f8')@np.asarray(bb).astype('f8') if rotation else reference
        value=M._peel_product(aa,bb,plan)
    result=np.asarray(value).copy(); outputs.append(result)
    err=result.astype('f8')-reference
    row=dict(rotation=rotation,flops=int(budget.flops_used),charged_operations=len(budget._op_log),wall_seconds=time.monotonic()-before,relative_error=float(np.linalg.norm(err)/np.linalg.norm(reference)),max_abs_error=float(np.max(np.abs(err))),rotation_rounding_relative=float(np.linalg.norm(transformed-reference)/np.linalg.norm(reference)),maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    results.append(row); print('ROW',json.dumps(row),flush=True)
    del result,value,err,transformed
    if rotation: del aa,bb,ap,am,bp,bm
    M._BH_HOLD.clear();gc.collect()
np.savez(D/'PAIR_ROTATION_OUTPUTS_v3.npz',reference=reference,baseline=outputs[0],rotation=outputs[1],rows=rows,cols=cols[:c15])
receipt=dict(prereg=prereg,rows=results,relative_error_recovery_pct=100*(1-results[1]['relative_error']/results[0]['relative_error']),extra_flops=results[1]['flops']-results[0]['flops'],extra_bill_pct_of240GF=100*(results[1]['flops']-results[0]['flops'])/240e9,charged_operation_delta=results[1]['charged_operations']-results[0]['charged_operations'],seconds=time.monotonic()-start,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Saved full-size actual kernel. Error measured against exact FP64 contraction of current FP32 weighted operands. No endpoint accuracy claim; request proxy excludes dynamic alias handles and disposal.')
(D/'PAIR_ROTATION_NATIVE_v3.json').open('x').write(json.dumps(receipt,indent=2)+'\n'); print('DONE',json.dumps(receipt),flush=True)
