"""Actual-client exact IO view shape/stride/bidirectional alias and lifecycle proof."""
import gc,hashlib,json,sys,types
from pathlib import Path
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _connection
D=Path(__file__).resolve().parent;R=D.parents[1]
source=R/'astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v13.py';fragment=D/'fragment_mainread_io_reshape_v1.py.txt'
requests=[0];send=_connection.Connection.send_recv
def counted(self,raw):requests[0]+=1;return send(self,raw)
_connection.Connection.send_recv=counted
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
M=types.ModuleType('mainread_reshape_candidate');M.__file__=str(source);sys.modules[M.__name__]=M
with flops.BudgetContext(flop_budget=2**50,quiet=True)as budget:
 exec(compile(source.read_text(),str(source),'exec'),M.__dict__)
 exec(compile(fragment.read_text(),str(fragment),'exec'),M.__dict__)
 env=M._ASTRA_SM40_ENV
 dims=tuple(max(f[i]*(1<<f[1])for f in M._ASTRA_SM40_FAMILIES)for i in (3,4,5))
 mr,mk,mc=dims
 env._sm40_outer=dict(A=fnp.empty((40,mr,mk),dtype=fnp.float32),B=fnp.empty((40,mk,mc),dtype=fnp.float32),C=fnp.empty((40,mr,mc),dtype=fnp.float32))
 for x in env._sm40_outer.values():fnp.copyto(x,0.)
 r0=requests[0]
 f0=int(budget.flops_used);count=M._mrio_prepare_reshapes()
 setup=dict(requests=requests[0]-r0,flops=int(budget.flops_used)-f0,keys=count)
 r0=requests[0];assert M._mrio_prepare_reshapes()==count;assert requests[0]==r0
 rows=[]
 for j,(side,L,m,n)in enumerate(M._MRIO_FIXED_KEYS):
  root=env._sm40_outer[('A','B','C')[side]];owner=root[:,:m,:n];u=1<<L
  parent=fnp.reshape(owner,(40,)+(2,)*L+(m//u,)+(2,)*L+(n//u,))
  r0=requests[0];view=M._mrio_view(env,fnp,owner,side,L,m,n,40);assert requests[0]==r0
  assert view.shape==parent.shape and view.strides==parent.strides and str(view.dtype)==str(parent.dtype)=='float32'
  tests=[]
  for hi in (False,True):
   idx=tuple((d-1 if hi else 0)for d in parent.shape[:-1])+(slice(None),)
   a=parent[idx];b=view[idx]
   fnp.copyto(a,float(j+1));ba=a._fetch_data()[0];bb=b._fetch_data()[0];assert ba==bb
   fnp.copyto(b,float(-j-1));ba=a._fetch_data()[0];bb=b._fetch_data()[0];assert ba==bb
   tests.append(hashlib.sha256(ba).hexdigest())
  rows.append(dict(key=[side,L,m,n],shape=list(view.shape),strides=list(view.strides),exact=True,bidirectional_alias=True,sample_hashes=tests))
 # Owner identity must reject old cached views even before explicit clear.
 old=env._sm40_outer;env._sm40_outer={}
 replacement=fnp.empty((40,64,64),dtype=fnp.float32);fnp.copyto(replacement,17.)
 r0=requests[0];fallback=M._mrio_view(env,fnp,replacement,2,5,64,64,40);fallback_requests=requests[0]-r0
 assert fallback_requests==1 and fallback.shape==(40,)+(2,)*5+(2,)+(2,)*5+(2,)
 assert fallback[(0,)*12+(slice(None),)]._fetch_data()[0]==replacement[0,0,:2]._fetch_data()[0]
 env._sm40_outer=old;M._astra_sm40_clear()
 assert env._mrio_views=={} and env._mrio_owner is None and not old
 # Rebind creates a fresh owner marker and fixed cache, then clear releases both.
 env._sm40_outer=dict(A=fnp.empty((40,mr,mk),dtype=fnp.float32),B=fnp.empty((40,mk,mc),dtype=fnp.float32),C=fnp.empty((40,mr,mc),dtype=fnp.float32))
 M._mrio_prepare_reshapes();assert env._mrio_owner is env._sm40_outer and len(env._mrio_views)==64
 M._astra_sm40_clear();assert not env._mrio_views and env._mrio_owner is None
out=dict(status='PASS',models=0,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(),owner_dims=dims,setup=setup,rows=rows,identity_fallback_requests=fallback_requests,repeat_setup_zero_requests=True,clear_rebind_pass=True,proof='Exact public-client shape/stride for every fixed key, bidirectional byte alias at low/high corners; same-origin prefix crop establishes whole-view address equivalence. Owner mismatch takes original public reshape; clear releases all aliases and owner marker.')
(D/'CLIENT_TINY_v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'}),flush=True)
