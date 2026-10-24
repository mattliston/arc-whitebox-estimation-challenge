from pathlib import Path
import ast,hashlib,json,os,resource,sys,time,types
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;START=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((D/'SOURCE_MANIFEST_HELD_v1.json').read_text())
S=Path(manifest['candidate_path']);P=Path(manifest['parent_path']);F=Path(manifest['fragment_path'])
for key in('candidate','parent','fragment'):assert sha(manifest[key+'_path'])==manifest[key+'_sha256']
assert os.getpriority(os.PRIO_PROCESS,0)==10
(D/'TINY_PROCESS_v1.json').open('x').write(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgid(0),source_sha256=sha(__file__)))+'\n')
ps=P.read_text();cs=S.read_text()
def nodes(source):
 return {n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
pn,cn=nodes(ps),nodes(cs)
changed=[]
for name,n in pn.items():
 if ast.dump(n,include_attributes=False)!=ast.dump(cn[name],include_attributes=False):changed.append(name)
assert changed==['_flms_cov_star','_dc_cov_star'],changed
for name in pn:assert pn[name].lineno==cn[name].lineno
assert ast.dump(pn['_dc_cov_diagonal'])==ast.dump(cn['_dc_cov_diagonal'])
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
M=types.ModuleType('cct_tiny');M.__file__=str(S);sys.modules[M.__name__]=M
with fs.BudgetContext(flop_budget=2**60,quiet=True):exec(compile(cs,str(S),'exec'),M.__dict__)
assert all(M.__dict__.get(k)is v for k,v in M._ASTRA_DENSE_CONTEXT)
# Retain exact parent function bodies for OFF and unchanged STAR comparisons.
for name in('_flms_cov_star','_dc_cov_star'):
 node=pn[name];clone=ast.parse(ast.get_source_segment(ps,node)).body[0]
 clone.name='_parent'+name
 exec(compile(ast.fix_missing_locations(ast.Module(body=[clone],type_ignores=[])),'parent_extract','exec'),M.__dict__)
M._CCT_ACTIVE[0]=True;M._W5_PORT_SCOPE={'layer':7}
rng=np.random.default_rng(20261015);cases=[];route=[]
# Tiny actual FNP products isolate the unchanged branch call graph.
originals={k:M.__dict__[k] for k in ('_mm','_sym_upper','_psdm_sym_upper','_dcv_tiles','_dcv_roster','_ss_factors','_astra_star_scaled_product')}
def mm(a,b):route.append('first');return a@b
def sym(a,b):route.append('sym');return a@b
def psdm(a,b,m):route.append('psdm');return a@b
def tiles(a,b,m):route.append('tiles');return a@b
def factors(wt,wt2,A,h):
 # Constant test STAR factors, shared by candidate and exact extracted parent.
 return fnp.sum(wt,axis=1)[:,None],fnp.sum(wt2,axis=1)[:,None]
def star(T,P,h):return (T@P.T)/3
M._mm=mm;M._sym_upper=sym;M._psdm_sym_upper=psdm;M._dcv_tiles=tiles;M._ss_factors=factors;M._astra_star_scaled_product=star
def run_case(label,n,out,m,mode,full=False):
 w=rng.normal(size=(n,out)).astype('f')*.1
 Z=rng.normal(size=(n,5)).astype('f')*.2;C=(Z@Z.T+np.eye(n,dtype='f')*.05).astype('f')
 A=rng.normal(size=(n,n)).astype('f')*.01
 order=rng.permutation(n).astype('int64');live=order[:m]
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  ww,cc,aa=[fnp.asarray(x)for x in(w,C,A)];rr=fnp.asarray(order if mode=='pad'else live);gate=fnp.ones(n,dtype=fnp.float32)
 M._DCV_SYM=mode;M._DCV_STATS['m']=m;M._dcv_roster=lambda *args:None if full else rr
 M._CCT_ON=False;route.clear()
 with fs.BudgetContext(flop_budget=2**60,quiet=True):off=M._dc_cov_star(ww,cc,aa,gate,.01,retain32=True)
 offnp=[np.asarray(x).copy()for x in off];offroute=list(route);route.clear()
 with fs.BudgetContext(flop_budget=2**60,quiet=True):parent=M._parent_dc_cov_star(ww,cc,aa,gate,.01,retain32=True)
 assert all(np.array_equal(a,np.asarray(b))for a,b in zip(offnp,parent))
 M._CCT_ON=True;route.clear();M._CCT_STATS=dict(calls=0,layers=[],rows=[],retired=False)
 with fs.BudgetContext(flop_budget=2**60,quiet=True):on=M._dc_cov_star(ww,cc,aa,gate,.01,retain32=True)
 got=[np.asarray(x).copy()for x in on]
 assert route==offroute,(label,route,offroute)
 keep=np.arange(n)if full else live
 ref=w[keep].astype('d').T@C[np.ix_(keep,keep)].astype('d')@w[keep].astype('d')
 assert np.allclose(got[0],ref,rtol=2e-5,atol=2e-5)
 assert np.array_equal(got[0],got[2].astype('d'))
 assert np.array_equal(got[1],offnp[1])and np.array_equal(got[3],offnp[3])
 assert np.array_equal(np.asarray(ww),w)and np.array_equal(np.asarray(cc),C)and np.array_equal(np.asarray(aa),A)
 assert M._CCT_STATS==dict(calls=1,layers=[7],rows=[len(keep)],retired=False)
 cases.append(dict(case=label,shape=[n,out],retained=len(keep),routes=list(route),max_error=float(np.max(np.abs(got[0]-ref))),OFF_parent_byte_exact=True,STAR_byte_exact=True,Cpre_exact_widening=True,inputs_unchanged=True))
run_case('full_roster_fallback',8,6,8,'pad',True)
run_case('padded_fallback',8,6,4,'pad')
run_case('direct_tiles',8,6,4,'tiles')
run_case('direct_sym',8,6,4,'direct')
run_case('specialized_psdm',1024,1024,64,'pad')
M.__dict__.update(originals)
# Scope guards are zero-FLOP identity; no stale _PHYS_LAYER dependency.
with fs.BudgetContext(flop_budget=2**60,quiet=True):small=fnp.eye(4,dtype=fnp.float32);weight=fnp.ones((4,3),dtype=fnp.float32);empty=fnp.empty((0,0),dtype=fnp.float32)
for on,active,scope,C in((False,True,{'layer':7},small),(True,False,{'layer':7},small),(True,True,None,small),(True,True,{'layer':0},small),(True,True,{'layer':15},small),(True,True,{'layer':7},empty)):
 M._CCT_ON=on;M._CCT_ACTIVE[0]=active;M._W5_PORT_SCOPE=scope
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as b:
  x,aux=M._cct_center(C,weight,M._E2Hold());out=M._cct_restore(x,aux,M._E2Hold())
 assert out is C and aux is None and b.flops_used==0 and not b._op_log
# Literal production helper tariff, conservative full1024 all14hidden upper.
M._CCT_ON=True;M._CCT_ACTIVE[0]=True;M._W5_PORT_SCOPE={'layer':7}
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 cc=fnp.asarray(rng.normal(size=(1024,1024)).astype('f'));ww=fnp.asarray(rng.normal(size=(1024,1024)).astype('f'));cov=fnp.asarray(rng.normal(size=(1024,1024)).astype('f'))
with fs.BudgetContext(flop_budget=2**60,quiet=True)as b:
 hold=M._E2Hold();_,aux=M._cct_center(cc,ww,hold);out=M._cct_restore(cov,aux,hold)
tariff=dict(flops_per_full_hidden=int(b.flops_used),ops_per_hidden=len(b._op_log),full14_upper_flops=14*int(b.flops_used),full14_upper_percent=1400*int(b.flops_used)/242136236203)
assert tariff['full14_upper_percent']<=.065
# Outer ownership cleanup, both normal and exceptional paths.
marker=object()
for on in(False,True):
 for throws in(False,True):
  M._CCT_ACTIVE[0]=False;M._CCT_ON=on
  def fake(*args,**kw):
   assert M._CCT_ACTIVE[0]is on
   if throws:raise RuntimeError('test')
   return marker
  M._CCT_PARENT_PREDICT=fake
  try:got=M._cct_predict(None)
  except RuntimeError:assert throws
  else:assert not throws and got is marker
  assert M._CCT_ACTIVE==[False]and M._CCT_STATS['retired']
r=dict(status='PASS',candidate_sha256=sha(S),source_sha256=sha(__file__),parent_sha256=sha(P),import_context_identity_pass=True,only_two_existing_functions_changed=True,all_existing_lines_preserved=True,L0_and_terminal_diagonal_source_exact=True,cases=cases,OFF_nonhidden_m0_identity_zero_ops=True,normal_exception_cleanup=True,tariff=tariff,new_setup_keys=0,new_einsum_signatures=0,seconds=time.monotonic()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,full_model_calls=0)
p=D/'TINY_AND_TARIFF_PASS_v1.json';p.open('x').write(json.dumps(r,indent=2)+'\n')
print(json.dumps(r));print('receipt_sha256',sha(p))

