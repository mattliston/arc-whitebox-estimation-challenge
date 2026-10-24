"""Existing saved vectors + bounded synthetic FNP; no models/captures/outcomes."""
import ast,hashlib,json,os,sys,types
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;R=D.parent/'astra_rtt2_request_repair_source_v1';OLD=D.parent/'astra_read_tail_rank2_native_v1'
assert all(os.environ.get(k)=='1'for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
class Hold(list):
 def __call__(self,x):self.append(x);return x
base=(R/'fragment_basis_norm_stack_HELD_v2.py').read_text();frag=(D/'fragment_prefix_power_QR_HELD_v2.py').read_text()
def env(lib):
 ns={'fnp':lib};exec(base,ns);ns['_RTP_REPAIRED_BASIS']=ns['_rtt_basis']
 funcs=[n for n in ast.parse(frag).body if isinstance(n,ast.FunctionDef)and n.name in ('_rtp_qr_basis','_rtp_refine')]
 exec(compile(ast.Module(body=funcs,type_ignores=[]),'private_QR_functions','exec'),ns);return ns
ns=env(fnp);nn=env(np);rng=np.random.default_rng(7261048)
cap=D.parent/'astra_read_column_tail_rank2_v1/run_capture_train16_v1/native_16_capture_capture.npz';assert hashlib.sha256(cap.read_bytes()).hexdigest()=='da2f504b55cab21b5a60bfecc7c553a866539839d2b7672d6bcfc8917da54a97'
meta=json.loads((cap.parent/'native_16_capture.json').read_text())['capture_metadata'];meta={a['layer']:a for a in meta}
cases=[]
with np.load(cap,allow_pickle=False)as z:
 for l in range(2,15):cases.append((str(l),z[f'L{l}_response_v'],z[f'L{l}_response_mu']))
for dtype in ('f','d'):
 x=np.linspace(-1,1,32).astype(dtype);zero=np.zeros(32,dtype)
 cases.extend([(dtype+'_zero',zero,zero),(dtype+'_rank1',x,2*x),(dtype+'_firstzero',zero,x),(dtype+'_secondzero',x,zero),(dtype+'_signedzero',np.r_[np.ones(16),np.zeros(16)].astype(dtype),np.r_[np.ones(16),-np.zeros(16)].astype(dtype)),(dtype+'_near_dependent',x,x+np.eye(1,32,3,dtype=dtype)[0]*1e-14)])
rows=[];priced=[]
for name,v,mu in cases:
 with fs.BudgetContext(flop_budget=2**50,quiet=True):vf=fnp.asarray(v);mf=fnp.asarray(mu)
 before=[np.asarray(vf).tobytes(),np.asarray(mf).tobytes()];outputs=[];cost=[]
 for fn in (ns['_rtt_basis'],ns['_rtp_qr_basis']):
  with fs.BudgetContext(flop_budget=2**50,quiet=True)as b:out=fn(vf,mf,Hold())
  outputs.append(np.asarray(out).copy());cost.append([int(b.flops_used),len(b._op_log)])
 a,q=outputs;assert a.shape==q.shape
 # Projector difference is invariant to QR signs; rank-deficient fallback must be byte-exact.
 err=float(np.linalg.norm(a.astype('d')@a.astype('d').T-q.astype('d')@q.astype('d').T))
 if not name.isdigit():assert a.tobytes()==q.tobytes(),name
 assert err<2e-6,(name,err)
 assert before==[np.asarray(vf).tobytes(),np.asarray(mf).tobytes()]
 rows.append(dict(case=name,rank=q.shape[1],base=cost[0],QR=cost[1],projector_frobenius_difference=err,exact_bytes=a.tobytes()==q.tobytes(),input_unchanged=True))
 if name.isdigit():
  item=meta[int(name)];m,c,r=int(item['m']),int(item['c15']),int(item['old_rank'])
  # Geometry-only synthetic operands; never a true network prefix or scientific result.
  P=rng.normal(size=(m,c)).astype('f');I=rng.normal(size=(m,r)).astype('f');B=rng.normal(size=(c,r)).astype('f')
  with fs.BudgetContext(flop_budget=2**50,quiet=True):fp=fnp.asarray(P);fq=fnp.asarray(q);fi=fnp.asarray(I);fb=fnp.asarray(B)
  with fs.BudgetContext(flop_budget=2**50,quiet=True)as b:out,qr,loss=ns['_rtp_refine'](fp,fq,fi if r else None,fb if r else None,Hold())
  assert qr==2 and not loss
  # B-prefix gather may miss a cache; conservatively charge one 4*c*r gather.
  gather_flops=4*c*r if r else 0;gather_ops=int(bool(r))
  priced.append(dict(layer=int(name),m=m,c15=c,old_rank=r,base_basis=cost[0],QR_basis=cost[1],refine=[int(b.flops_used),len(b._op_log)],cold_prefix_gather=[gather_flops,gather_ops],increment_vs_e406=[cost[1][0]+int(b.flops_used)+gather_flops-cost[0][0],cost[1][1]+len(b._op_log)+gather_ops-cost[0][1]],max_fixture_input_bytes=sum(x.nbytes for x in(P,I,B,q))))
# Exact annihilation and rank1 input preserve original space, never arbitrary QR completion.
Q=np.eye(32,2,dtype='f');P=np.zeros((32,16),dtype='f');o,rank,loss=nn['_rtp_refine'](P,Q,None,None,Hold());assert o is Q and rank==0 and loss
q=Q[:,:1];o,rank,loss=nn['_rtp_refine'](P,q,None,None,Hold());assert o is q and loss
# Same integrated synthetic dispatch as original proof, with exact repaired OFF delegation.
REPAIRDIR=R
repaired=(OLD/'fragment_rtt2_HELD_v1.py').read_text()+'\n'+(REPAIRDIR/'fragment_rtt2_request_repair_HELD_v1.py').read_text()+'\n'+base
private_dir=D
donor=(OLD/'check_rtt2_v1.py').read_text().split('\nns,calls,state=make(np);')[0]
donor=donor.replace('D=Path(__file__).resolve().parent','D=OLD')
exec(compile(donor,'existing_tiny_setup','exec'),globals());D=private_dir
outs=[];integrated=[]
for mode in ('BASE','OFF','ON'):
 fragment=repaired+(''if mode=='BASE'else '\n'+frag)
 with fs.BudgetContext(flop_budget=2**40,quiet=True):
  e,calls,state=make(fnp);lf=fnp.asarray(L);rf=fnp.asarray(R);la=fnp.asarray(lam);ids=fnp.arange(n)
 if mode!='BASE':e['_RTP_ON']=mode=='ON'
 with fs.BudgetContext(flop_budget=2**40,quiet=True)as b:out=e['_rtt_read'](lf,rf,la,ids,state)
 outs.append(np.asarray(out).copy())
 assert len(calls)==1 and e['_RTT_CONTEXT']==[None] and not e['_RTT_STATS']['reads'][-1]['forced_d7']
 assert e['_YR7_SHAPES']==e['_RTT_FIXED_YR7_SHAPES']
 integrated.append(dict(mode=mode,flops=int(b.flops_used),requests=len(b._op_log),old_inner_calls=1,forced_d7=False))
assert outs[0].tobytes()==outs[1].tobytes() and integrated[0]['flops']==integrated[1]['flops'] and integrated[0]['requests']==integrated[1]['requests']
assert outs[0][:,:32].tobytes()==outs[2][:,:32].tobytes()
e['_RTT_ACTIVE']=False
try:e['Estimator']().predict(fail=True)
except RuntimeError:pass
else:raise AssertionError('exception lost')
assert not e['_RTT_ACTIVE']and e['_RTT_CONTEXT']==[None]and e['_RTT_STATS']['retired']
# Full imports only: never instantiate an Estimator or run setup/predict.
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
imports=[]
for mode in ('OFF','ON'):
 path=D/f'candidate_prefix_power_QR_{mode}_HELD_v2.py';mod=types.ModuleType('private_rtp_qr_'+mode);mod.__file__=str(path);sys.modules[mod.__name__]=mod
 with fs.BudgetContext(flop_budget=2**60,quiet=True):exec(compile(path.read_bytes(),str(path),'exec'),mod.__dict__)
 assert mod._RTP_ON==(mode=='ON')and mod._RC_PARENT_READ is mod._rtt_read
 assert mod._RTT_CONTEXT==[None]and not mod._RTT_ACTIVE and mod._YR7_SHAPES==mod._RTT_FIXED_YR7_SHAPES
 assert all(mod.__dict__.get(k)is v for k,v in mod._ASTRA_DENSE_CONTEXT)
 imports.append(dict(mode=mode,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),aliases_and_contexts=True))
receipt=dict(status='PASS_REBASED_QR_SOURCE_TINY_SAVED_VECTORS',capture_sha256=hashlib.sha256(cap.read_bytes()).hexdigest(),basis_cases=rows,actual_FNP_geometry_prices=priced,synthetic_dispatch=integrated,imports=imports,exact_OFF_bytes_bill_requests=True,exact_prefix=True,rank_loss_retains_original_Q=True,no_model_calls=True,no_new_captures=True,no_accuracy_outcome_reads=True,scope='QR signs/reduction order differ; projector differences only. Existing capture lacks exact prefix/old factors, so power refinement is synthetic and geometry-priced, not an offline network accuracy result. Conservative prefix B-gather charged even if reused.',total_forecast_increment_vs_e406=[sum(x['increment_vs_e406'][0]for x in priced),sum(x['increment_vs_e406'][1]for x in priced)],max_basis_projector_F_error=max(x['projector_frobenius_difference']for x in rows))
with(D/'REPAIRED_QR_TINY_AND_TARIFF_astra_v1.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in receipt.items()if k not in ('basis_cases','actual_FNP_geometry_prices')}))
