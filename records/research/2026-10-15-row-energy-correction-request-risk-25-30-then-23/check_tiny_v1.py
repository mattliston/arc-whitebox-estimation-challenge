from pathlib import Path
import ast,hashlib,json,sys,types,os,random,struct,gc,time,resource
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;remote='--remote'in sys.argv
assert os.getpriority(os.PRIO_PROCESS,0)==10
assert all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
cfg=json.loads((D/'SOURCE_HELD_v1.json').read_text());modules=[];counts=dict(requests=0,free_trips=0,freed=0)
if remote:
 from flopscope import _connection
 send=_connection.Connection.send_recv;drain=_connection.drain_pending
 def counted(self,raw):counts['requests']+=1;return send(self,raw)
 def drained():
  got=drain()
  if got:counts['free_trips']+=1;counts['freed']+=len(got)
  return got
 _connection.Connection.send_recv=counted;_connection.drain_pending=drained
class Hold(list):
 def __call__(self,x):self.append(x);return x
for role in ('parent','candidate'):
 p=Path(cfg[role]['path']);s=p.read_text();assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg[role]['sha256'];t=ast.parse(s)
 funcs={n.name:ast.get_source_segment(s,n)for n in t.body if isinstance(n,ast.FunctionDef)and n.name in ('_uvp_inner','_uvp_recon','_astra_bcc_uvrecon')}
 env=dict(fnp=fnp,_BH_HOLD=[],_E2Hold=Hold,_ubx_product=lambda a,b,*rest:fnp.matmul(a,b),_ubc_product=lambda a,b,*rest:fnp.matmul(a,b))
 for text in funcs.values():exec(compile(text,str(p)+':selected','exec'),env)
 modules.append(env)
rng=random.Random(20261015)
def matrix(m,n):return [[rng.uniform(-1,1)for j in range(n)]for i in range(m)]
def bits(x):
 vals=x.tolist();flat=[a for row in vals for a in row];return struct.pack('<'+'f'*len(flat),*flat)
rows=[];start=time.monotonic()
with fs.BudgetContext(flop_budget=2**50,quiet=True) as budget:
 for count in (1,3,6,9):
  widths=[2+i%3 for i in range(count)];blocks=[];off=0
  for width in widths:blocks.append((off,width));off+=2*width
  p=sum(widths);m,r,ru=17,8,4
  Fu=fnp.asarray(matrix(ru,p),dtype=fnp.float32);Fv=fnp.asarray(matrix(r,p),dtype=fnp.float32);left=fnp.asarray(matrix(m,2*p),dtype=fnp.float32);lam=fnp.asarray([(-1 if j%2 else 1)*(.2+j*.001)for j in range(2*p)],dtype=fnp.float32);U=fnp.asarray(matrix(m,r),dtype=fnp.float32)
  uv=(Fu,Fv,blocks,ru);before=[bits(x)for x in (Fu,Fv,left,U)];one={}
  for name in ('_uvp_inner','_uvp_recon','_astra_bcc_uvrecon'):
   outs=[];charges=[]
   for role,env in zip(('parent','candidate'),modules):
    out=fnp.empty((m,2*p),dtype=fnp.float32)if name=='_astra_bcc_uvrecon'else None
    env['_BH_HOLD'].clear();gc.collect();flush=fnp.empty((0,),dtype=fnp.float32);counts.update(requests=0,free_trips=0,freed=0)
    prior_flops=int(budget.flops_used)
    answer=env[name](left,lam,uv)if name=='_uvp_inner'else(env[name](U,uv)if name=='_uvp_recon'else env[name](U,uv,out))
    flush=fnp.empty((0,),dtype=fnp.float32);measured=counts.copy();measured['total_rpc']=measured['requests']+measured['free_trips']
    measured['flops']=int(budget.flops_used)-prior_flops
    assert not env['_BH_HOLD'];assert out is None or answer is out
    outs.append(bits(answer));charges.append(measured)
    del answer,out
   assert outs[0]==outs[1],(count,name,'bytes');assert charges[0]['flops']==charges[1]['flops'],(count,name,'bill')
   one[name]=dict(parent=charges[0],candidate=charges[1],RPC_saved=charges[0]['total_rpc']-charges[1]['total_rpc'],bytes_equal=True,flops_equal=True)
  assert before==[bits(x)for x in (Fu,Fv,left,U)]
  rows.append(dict(cohorts=count,functions=one))
forecast=sum(6*next(z for z in rows if z['cohorts']==b)['functions'][name]['RPC_saved']for b in (3,6)for name in ('_uvp_inner','_uvp_recon','_astra_bcc_uvrecon'))
result=dict(status='PASS',remote_client=remote,candidate_sha256=cfg['candidate']['sha256'],parent_sha256=cfg['parent']['sha256'],rows=rows,hypothetical_schedule_RPC_saving=forecast,schedule='Not an actual call census; previous static forecast overestimated. Per-call measured counts are authoritative; production must price total.',original_product_functions_mocked_by_identical_fnp_matmul=True,scope='Actual FNP selected-function byte/FLOP proof and, if remote, actual numpy-free-client request/free-trip counts; no setup, predict, model, saved capture or truth.',seconds=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
p=D/('REMOTE_TINY_v1.json'if remote else'ACTUAL_FNP_TINY_v1.json')
with p.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result),flush=True)
