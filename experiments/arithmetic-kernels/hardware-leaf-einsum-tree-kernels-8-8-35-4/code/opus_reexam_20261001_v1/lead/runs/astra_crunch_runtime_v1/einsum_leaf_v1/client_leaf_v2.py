"""Private isolated Bini10 qualification; candidate uses public FNP, no model."""
import gc, hashlib, json, os, signal, sys, time, types
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[3]
sys.path.insert(0,str(O/'release'))
import rusage
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _connection, _dispatch
requests=[0];send=_connection.Connection.send_recv
def counted_send(self,raw):
 requests[0]+=1
 return send(self,raw)
_connection.Connection.send_recv=counted_send
_dispatch.__dict__.update(_release_raw=rusage.raw,_release_depth=[0],_release_acc=[0,0],_release_span=_dispatch._counted_span)
exec('''
@contextmanager
def _counted_span():
 with _release_span():
  top=_release_depth[0]==0
  if top:c0,i0=_release_raw()
  _release_depth[0]+=1
  try:yield
  finally:
   _release_depth[0]-=1
   if top:
    c1,i1=_release_raw();_release_acc[0]+=c1-c0;_release_acc[1]+=i1-i0
''',_dispatch.__dict__)
acc=_dispatch._release_acc
out=Path(sys.argv[1]);assert not out.exists()
factory=Path(sys.argv[2]);source=D.parent/'s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_runtime_v8.py'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
M=types.ModuleType('isolated_candidate');M.__file__=str(source);sys.modules[M.__name__]=M
exec(compile(source.read_text(),str(source),'exec'),M.__dict__)

def server_mark():
 marks=Path(os.environ['RELEASE_SERVER_MARKS']);before=marks.stat().st_size if marks.exists()else 0
 os.kill(int(os.environ['RELEASE_SERVER_PID']),signal.SIGUSR1)
 for _ in range(4000):
  if marks.exists()and marks.stat().st_size>before:break
  time.sleep(.0005)
 else:raise RuntimeError('server mark failed')
 return json.loads(marks.read_text().splitlines()[-1])
def measure(label,call):
 m0=server_mark();r0=requests[0];d0=acc[:];c0,i0=rusage.raw();t0=time.perf_counter()
 with flops.BudgetContext(flop_budget=2**42,quiet=True,namespace=label)as budget:result=call()
 wall=time.perf_counter()-t0;c1,i1=rusage.raw();r1=requests[0];d1=acc[:];m1=server_mark()
 row=dict(label=label,wall_s=wall,requests=r1-r0,client_instructions=i1-i0,residual_instructions=i1-i0-(d1[1]-d0[1]),client_cycles=c1-c0,residual_cycles=c1-c0-(d1[0]-d0[0]),summary=budget.summary_dict(),server_instructions=m1['instructions']-m0['instructions'],server_kernel_s=m1['kernel_s']-m0['kernel_s'],server_numpy_ops={k:[v[j]-m0['numpy_ops'].get(k,[0,0,0,0])[j] for j in range(4)]for k,v in m1['numpy_ops'].items()},server_peak_rss_bytes=m1['max_rss_bytes'],server_peak_footprint_bytes=m1['footprint_max'])
 return result,row


import array,math,random
rows=[];checks=[];rng=random.Random(20261017)
with flops.BudgetContext(flop_budget=2**44,quiet=True):
 arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32)
cases=[('push_p10_cropped',('peel6_p10_5_5_26',6,10,5,5,26),(5,5,21)),('push_p10_full',('peel6_p10_5_5_26',6,10,5,5,26),(5,5,26)),('read_p20',('peel6_p20_5_11_5',6,20,5,11,5),(5,11,5)),('read_p10_cropped',('peel6_p10_5_21_5',6,10,5,21,5),(5,16,5)),('read_p10_full',('peel6_p10_5_21_5',6,10,5,21,5),(5,21,5))]
for label,family,(r,k,c) in cases:
 with flops.BudgetContext(flop_budget=2**44,quiet=True):
  frame=M._astra_sm40_bind_family(fnp,M,arena,family);TA,TB,TC=(side['T'] for side in frame['sides'])
  TA=TA[:3,...,:r,:k];TB=TB[:3,...,:k,:c];TC=TC[:3,...,:r,:c]
  av=array.array('f',(rng.uniform(-1,1)*[1,.01,10][i%3] for i in range(r*k)));bv=array.array('f',(rng.uniform(-1,1)*[1,10,.01][i%3] for i in range(k*c)))
  aa=fnp.reshape(fnp.frombuffer(av.tobytes(),dtype=fnp.float32),(r,k));bb=fnp.reshape(fnp.frombuffer(bv.tobytes(),dtype=fnp.float32),(k,c))
  fnp.copyto(TA,aa);fnp.copyto(TB,bb);fnp.copyto(TC,0.)
 ref=[sum(float(av[i*k+t])*float(bv[t*c+j])for t in range(k))for i in range(r)for j in range(c)]
 vals={}
 for rep,variant in enumerate(('matmul','einsum','einsum','matmul','matmul','einsum')):
  fn=(lambda:fnp.matmul(TA,TB,out=TC))if variant=='matmul'else(lambda:fnp.einsum('...ik,...kj->...ij',TA,TB,out=TC))
  y,row=measure(label+'_'+variant+'_'+str(rep),fn);row.update(case=label,variant=variant,rep=rep,family=family,leaf=(r,k,c),leaf_shape=list(TA.shape));rows.append(row)
  with flops.BudgetContext(flop_budget=2**44,quiet=True):
   sample=TC[(0,)*6+(0,)];data=sample._fetch_data()[0]
  values=array.array('f');values.frombytes(data);vals[variant]=list(values)
  row['rel_rmse']=math.sqrt(sum((float(a)-b)**2 for a,b in zip(values,ref))/sum(b*b for b in ref));row['max_abs']=max(abs(float(a)-b)for a,b in zip(values,ref));row['sha256']=hashlib.sha256(data).hexdigest()
  print(json.dumps({q:row[q]for q in ('case','variant','rep','wall_s','server_instructions','server_kernel_s','requests','rel_rmse','max_abs')}),flush=True)
  del y,sample
 checks.append(dict(case=label,byte_equal=vals['matmul']==vals['einsum'],max_abs_difference=max(abs(a-b)for a,b in zip(vals['matmul'],vals['einsum'])),flops_equal=len({x['summary']['flops_used']for x in rows if x['case']==label})==1))
 del frame,TA,TB,TC,aa,bb;gc.collect()
result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,checks=checks,models=0,leading_batch_prefix=3,full_leading_extent=7,geometry='Exact production strides/P/leaf dims; 3/7 outermost batch prefix bounds physical memory')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(checks),flush=True)
