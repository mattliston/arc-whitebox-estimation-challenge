"""Two shape-only public0.12.3 tariff checks; no estimator or numerical matrix."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,sys,time,hashlib,resource,gc
D=Path(__file__).resolve().parent;B=D.parent;R=B.parent
source=R/'astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf'
pre=R/'astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py'
s=pre.read_text().split('def make_bini')[0].replace("p.SOURCE=O/'lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py'",'p.SOURCE=Path('+repr(str(source))+')')
s=s.replace("    if name in ('copy','astype')", "    if name in ('put','put_along_axis'):\n        p.CALLS[name]+=1;return None\n    if name in ('copy','astype')")
ns={'__file__':str(pre),'__name__':'fusion_metadata'};exec(compile(s,str(pre),'exec'),ns)
M,fn,fs,arena,arr,measure=(ns[k]for k in('M','fn','fs','arena','arr','measure'))
assert str(fs.__version__).startswith('0.12.3')
# Same certified packed classification and eager sides as prediction. No estimator setup.
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 def bind(key):
  kind,L,r,c=key;e=M._ipt_side(kind,L,r,c);slot=e[2]
  if type(slot)is list and slot[1]is None:M._l3o_pairs(slot,L)if kind==2 else M._l3_quad(slot,kind,L)
 M._LB_SETUP[0]=True;M._LVF_EAGER[0]=True
 for key in M._PB_SINGLE[:128]:bind(key)
 M._LB_SETUP[0]=False;M._LVF_EAGER[0]=False;M._lb_make_boxes();M._gcv_prepare(280)
 M._LVF_EAGER[0]=True
 for key in M._PB_SINGLE[128:192]:bind(key)
 M._LVF_EAGER[0]=False
case=int(sys.argv[1]);shapes=[(896,832,832,4576),(960,896,864,5504)];m,my,k,n=shapes[case]
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 A=arr((m,k));Ay=arr((my,k));At=arr((m-my,k));Bq=arr((k,384));B=arr((k,n));rows=fn.empty((my,1),dtype=fn.int64);tail=fn.empty((m-my,1),dtype=fn.int64)
assert (m,k,384)not in M._SM12_RULE and (m-my,k,384)not in M._SM12_RULE
M._W5_PORT_SCOPE=None
# Originalunioncalls chosenhere are unpeeled; _mm is the exact parent fallback.
def baseline():
 U=M._shared_old384_mm(A,Bq)
 Y=M._mm(Ay,B)
 return U,Y
def fused():
 rhs=fn.concatenate((B,Bq),axis=1) # conservative: additionalconcat charged althoughexistingRHSconcatcanabsorbit
 Y=M._mm(Ay,rhs)
 Ut=M._shared_old384_mm(At,Bq)
 U=fn.empty((m,384),dtype=fn.float32)
 fn.put_along_axis(U,rows,Y[:,n:],axis=0)
 fn.put_along_axis(U,tail,Ut,axis=0)
 return U,Y[:,:n]
measure(baseline);b=measure(baseline)
measure(fused);f=measure(fused)
r=dict(status='METADATA_ONLY_ACTUAL0123_TARIFF',shape=shapes[case],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sdk=fs.__version__,base=b,fused=f,gain_flops=b['flops']-f['flops'],requests_saved=b['requests']-f['requests'],maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,models=0,servers=0,numerical_matrix_operations=0,limitations=['Actual public tariff, virtualzero-stride backend; not numerical parity or wall.','Warm algorithm views; coldownprebind and freesnotincluded.','Baseline U andyoungA are alreadygathered; fusiontailgather couldadd~3calls/64*k copying.','Additional RHSconcat conservativelycharged; direct existingRHSconcat integration mayreducecopycost.','Only currentlyunpeeled unioncalls considered, so no lostcompact10peelcredit.'])
out=D/f'FUSION_METADATA_v1_{case}.json';out.open('x').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
