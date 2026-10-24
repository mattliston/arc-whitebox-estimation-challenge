"""Three fixed routes: real0.12.3 tariff, virtual backend, no server/model/matmul."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,resource,time,hashlib
HERE=Path(__file__).resolve().parent; B=HERE.parents[1]; R=B.parent
source=R/'astra_oct18_accuracy_v1/l7_rank320_current_v1/candidate_95c_L7rank320_HELD_v1.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='a87daa2e30ace2366afc8392eb7b6028e6b399248da72e33bde076d1d1d0b531'
prefix=(B/'price_peel_deunion_v1.py').read_text().split('u=json.loads')[0]
prefix=prefix.replace("D=Path(__file__).resolve().parent;R=D.parent;t0=time.monotonic()",'D=Path('+repr(str(B))+');R=D.parent;t0=time.monotonic()')
prefix=prefix.replace("source=R/'opus_fix336687_v1/candidate_v61j.py'",'source=Path('+repr(str(source))+')')
prefix=prefix.replace(' M._smx_bind();M._peel_bind_read_io()', '''
 M._SMX_FAMILIES={k:v for k,v in M._SMX_FAMILIES.items() if k=='sm12_e1_L5P40'}
 M._PEEL_PUSH_PLANS={k:v for k,v in M._SM12_PLANS.items() if v[1]=='sm12_e1_L5P40'}
 M._PEEL_READ_PLANS={}
 M._smx_bind()
''')
exec(compile(prefix,str(B/'price_peel_deunion_v1.py'),'exec'),globals())
assert str(fs.__version__).startswith('0.12.3')

exec(compile((HERE.parent/'fusion_strip5_v1/fragment_v1.py').read_text(),'<strip>','exec'),M.__dict__)
exec(compile((HERE.parent/'fusion_strip5_sym_v1/fragment_v1.py').read_text(),'<sym>','exec'),M.__dict__)
rows=[]
for inner in (896,960):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  A=arr((1024,inner));B0=arr((inner,1024))
 base=lambda:M._O18_MARGIN_PARENT_PSDM(A,B0,inner)
 alt=lambda:M._o18_margin_two_strip(A,B0)
 measure(base);old=measure(base);measure(alt);new=measure(alt)
 rows.append(dict(kind='symmetric_strip',inner=inner,base=old,new=new,gain_flops=old['flops']-new['flops'],saved_requests=old['requests']-new['requests']))
for rank in (384,320):
 m,my,k,n=896,832,896,9504
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  A=arr((m,k));Ay=arr((my,k));At=arr((64,k));Bq=arr((k,rank));Y=arr((k,n));inds=fn.empty((my,1),dtype=fn.int64);tail=fn.empty((64,1),dtype=fn.int64)
 M._W5_PORT_SCOPE=None
 def base():return M._shared_old384_mm(A,Bq),M._mm(Ay,Y)
 def alt():
  rhs=fn.concatenate((Y,Bq),axis=1);v=M._mm(Ay,rhs)
  kept=fn.copy(v[:,n:]);young=fn.copy(v[:,:n])
  u=fn.empty((m,rank),dtype=fn.float32);ut=M._shared_old384_mm(At,Bq)
  fn.put_along_axis(u,inds,kept,axis=0);fn.put_along_axis(u,tail,ut,axis=0)
  return u,young
 measure(base);old=measure(base);measure(alt);new=measure(alt)
 rows.append(dict(kind='fusion',shape=[m,my,k,n,rank],base=old,new=new,gain_flops=old['flops']-new['flops'],saved_requests=old['requests']-new['requests']))
report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sdk=fs.__version__,rows=rows,models=0,servers=0,numeric_matrix_operations=0,seconds=time.monotonic()-t0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['Warm virtual tariff only; actual request/free/setup interactions require native.','Fusion includes conservative extra concat and whole young copy; actual newborn already copies young into owned answer.','No endpoint/rank-specific policy; one publicly observed complete shape supported.'])
assert report['peak_rss_bytes']<500_000_000
with (HERE/'PRICE_v1.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
for row in rows:print(json.dumps({k:v for k,v in row.items()if k not in ('base','new')}))
