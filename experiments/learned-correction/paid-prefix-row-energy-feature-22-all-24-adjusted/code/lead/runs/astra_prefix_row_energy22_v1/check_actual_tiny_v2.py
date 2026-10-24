from pathlib import Path
import hashlib,json,sys,types,os,time,resource
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;S=D/'candidate_prefix_energy22_HELD_v1.py';P=D.parent/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py';frag=(D/'fragment_prefix_energy22_HELD_v1.py').read_text();src=S.read_text()
assert src.endswith(frag)
assert src[:-len(frag)].replace('        _re22_capture(prefix,Q,rows,live_indices,c15,hold)\n','')==P.read_text()
assert os.getpriority(os.PRIO_PROCESS,0)==10
assert all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
M=types.ModuleType('prefix22_tiny');M.__file__=str(S);sys.modules[M.__name__]=M
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
with fs.BudgetContext(flop_budget=2**50,quiet=True):exec(compile(src,str(S),'exec'),M.__dict__)
assert all(M.__dict__.get(k)is v for k,v in M._ASTRA_DENSE_CONTEXT)
assert M._RE22_PARENT_FEATURES.__code__.co_firstlineno==201
rng=np.random.default_rng(20261016);cases=[];start=time.monotonic()
for rank,overlap,zero in ((2,True,False),(1,True,False),(0,True,False),(2,False,False),(2,True,True)):
 n,m,p,t=48,9,11,5;rows=np.asarray([16,2,10,4,0,6,14,8,12],dtype='i8');pc=np.asarray([9,2,5,4,8,1,7,0,3,10,6],dtype='i8')if overlap else np.arange(20,31,dtype='i8');tc=np.asarray([11,14,13,16,12],dtype='i8')if overlap else np.arange(32,37,dtype='i8')
 A=rng.normal(size=(m,p)).astype('f4');Q=np.linalg.qr(rng.normal(size=(m,rank)))[0].astype('f4');v=np.exp(rng.normal(size=n)).astype('f8')
 if zero:A[:]=0
 with fs.BudgetContext(flop_budget=2**50,quiet=True):aa,qq,rr,pp,tt,vv=[fnp.asarray(x)for x in(A,Q,rows,pc,tc,v)]
 before=[np.asarray(x).tobytes()for x in(aa,qq,rr,pp,tt,vv)]
 with fs.BudgetContext(flop_budget=2**50,quiet=True)as b:got=M._re22_energy(aa,qq,rr,pp,tt,vv,M._E2Hold())
 vf=v.astype('f4');res=A-Q@(Q.T@A);res[rows[:,None]==pc[None,:]]=0;norm=res/(vf[rows,None]*np.sqrt(vf[pc])[None,:]);expected=np.einsum('ij,ij->i',norm,norm,optimize=False)*(np.sum(1/vf[tc])/np.sum(1/vf[pc]));value=np.asarray(got).copy()
 assert np.allclose(value,expected,rtol=3e-6,atol=3e-6),(value,expected)
 assert before==[np.asarray(x).tobytes()for x in(aa,qq,rr,pp,tt,vv)]
 # Owned output remains detached after scratch-like source overwrite.
 with fs.BudgetContext(flop_budget=2**50,quiet=True):fnp.copyto(aa,fnp.float32(0));fnp.copyto(qq,fnp.float32(0))
 assert np.array_equal(np.asarray(got),value)
 x=rng.normal(size=(n,30)).astype('f4');x[:,22]=np.abs(x[:,22]);want=x.copy();want[rows,22]=np.sqrt(np.maximum(want[rows,22]**2+value,np.float32(0)))
 with fs.BudgetContext(flop_budget=2**50,quiet=True):xx=fnp.asarray(x)
 M._RE22_PARENT_FEATURES=lambda *a,**k:xx;M._RE22_ACTIVE[0]=True;M._RE22_PENDING[0]=(4,rr,got);M._RE22_STATS.update(reads=[],feature_layers=[])
 with fs.BudgetContext(flop_budget=2**50,quiet=True):result=M._re22_features(4)
 assert result is xx and np.array_equal(np.asarray(result),want)
 assert M._RE22_PENDING[0]is None and M._RE22_STATS['feature_layers']==[4]
 M._RE22_ACTIVE[0]=False
 cases.append(dict(rank=rank,overlap=overlap,zero=zero,max_error=float(np.max(abs(value-expected))),flops=int(b.flops_used),ops=len(b._op_log)))
# Synthetic active RTT dispatch: original read result stays byte identical.
n,m,k,c2,c15,old=64,32,8,64,32,4
L=rng.normal(size=(m,old+k)).astype('f4');B=rng.normal(size=(c2,k)).astype('f4');lam=np.r_[np.ones(old),np.ones(2),-np.ones(2),np.ones(2),-np.ones(2)].astype('f4');Q=np.linalg.qr(rng.normal(size=(m,2)))[0].astype('f4');inner=rng.normal(size=(m,2)).astype('f4');U=rng.normal(size=(n,2)).astype('f4')
with fs.BudgetContext(flop_budget=2**50,quiet=True):left,right,weights,indices,q,inn,uu,rows,mu,vc,g,sig=[fnp.asarray(x)for x in(L,B,lam,np.arange(n),Q,inner,U,np.arange(m),np.zeros(n),np.ones(n),np.r_[np.ones(c15),np.zeros(n-c15)],np.ones(n))]
state=dict(columns=old,pre_basis=uu,cohorts=[(0,old),(old,4),(old+4,4)])
M._RTT_ACTIVE=True;M._RS_LAYER[0]=7;M._E2_LAYER.update(ecut_rows=rows,mu=mu,vc=vc,g=g,s=sig)
M._rtp_qr_basis=lambda *a:q;M._rtp_refine=lambda *a:(q,2,False);M._xsg_rows=lambda site,a,idx:a[idx]
def parent_read(a,b,w,idx,state):
 if M._RTT_CONTEXT[0]is not None:M._RTT_CONTEXT[0].update(inner=inn,inner_calls=1)
 return a[:,old:]@(b*w[old:]).T+inn@uu[idx].T
M._RTT_PARENT_READ=parent_read;before=[np.asarray(x).tobytes()for x in(left,right,weights,uu)];outs=[]
for on in(False,True):
 M._RE22_ACTIVE[0]=on;M._RE22_PENDING[0]=None;M._RE22_STATS.update(reads=[],feature_layers=[])
 with fs.BudgetContext(flop_budget=2**50,quiet=True):out=M._rtt_read(left,right,weights,indices,state)
 outs.append(np.asarray(out).tobytes());assert M._RTT_CONTEXT[0]is None
 if on:
  assert M._RE22_PENDING[0]is not None and M._RE22_STATS['reads']==[dict(layer=7,m=32,prefix=32,tail=32,qrank=2)]
  with fs.BudgetContext(flop_budget=2**50,quiet=True):xx=fnp.ones((n,30),dtype=fnp.float32)
  M._RE22_PARENT_FEATURES=lambda *a,**k:xx
  with fs.BudgetContext(flop_budget=2**50,quiet=True):M._re22_features(7)
assert outs[0]==outs[1] and before==[np.asarray(x).tobytes()for x in(left,right,weights,uu)]
# Unchanged full-prefix route must create no energy or pending feature override.
M._RE22_ACTIVE[0]=True;M._RE22_PENDING[0]=None;M._RE22_STATS.update(reads=[],feature_layers=[])
with fs.BudgetContext(flop_budget=2**50,quiet=True):M._E2_LAYER['g']=fnp.ones(n,dtype=fnp.float64);M._rtt_read(left,right,weights,indices,state)
assert M._RE22_PENDING[0]is None and M._RE22_STATS['reads']==[]
M._re22_clear();marker=object();M._RE22_PARENT_FEATURES=lambda *a,**k:marker
with fs.BudgetContext(flop_budget=2**50,quiet=True)as off:
 assert M._re22_features(7)is marker
 M._re22_capture(None,None,None,None,None,None)
assert off.flops_used==0 and not off._op_log
for throwing in(False,True):
 def fake(*a,**kw):
  if throwing:M._RE22_PENDING[0]=object();raise RuntimeError('synthetic')
  return marker
 M._RE22_PARENT_PREDICT=fake
 try:r=M._re22_predict(None)
 except RuntimeError:assert throwing
 else:assert r is marker and not throwing
 assert M._RE22_ACTIVE==[False] and M._RE22_PENDING==[None] and M._RE22_STATS['retired']
out=dict(status='PASS',candidate_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),source_inverse_exact=True,all_original_context_bindings=True,original_feature_parent_line=201,cases=cases,read_ON_OFF_byte_identical=True,feature22_only_exact=True,other29_features_byte_identical=True,noncontiguous_unsorted_rosters=True,rank0_tested=True,no_tail_no_read_bypass=True,owned_energy_detached=True,source_arrays_unchanged=True,OFF_zero_ops=True,success_exception_cleanup=True,models=0,captures=0,seconds=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(D/'ACTUAL_FNP_TINY_v2.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
