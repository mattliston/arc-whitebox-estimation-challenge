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


import ast
ss=(HERE.parent/'fusion_strip5_sym_v1/candidate_95c_fusion_strip5_sym_HELD_v1.py').read_text()
current=(R/'astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py').read_text()
assert ss.startswith(current)
exec(compile(ss[len(current):],'<held margin>','exec'),M.__dict__)
node=[z for z in ast.parse(ss).body if isinstance(z,ast.FunctionDef)and z.name=='_astra_bcc_newborn_impl'][-1]
body=ast.get_source_segment(ss,node)
body=body.replace("(k, 384)","(k, int(ctx['basis'].shape[1]))").replace("(ctx['total'], 384)","(ctx['total'], int(ctx['basis'].shape[1]))")
exec(compile(body,'<rank generic newbody>','exec'),M.__dict__)
rows=[]
for rank in (384,320):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  w=arr((1024,832));wf=arr((1024,896));p0=arr((896,8768));prev=fn.empty((896,),dtype=fn.int64);selected=fn.empty((736,),dtype=fn.int64);base0=arr((1024,736));t=arr((736,));ss0=arr((736,));bq=arr((896,rank));A=arr((896,896));rr=fn.empty((832,),dtype=fn.int64);tt=fn.empty((64,),dtype=fn.int64)
  newborn=(8768,selected,base0,t,ss0)
  cache={(5154,id(w)):arr((896,832)),(5159,id(base0)):arr((896,736)),(5161,id(w)):arr((736,832))}
 def xrows(code,value,index):
  key=(code,id(value))
  if key not in cache:cache[key]=arr((int(index.shape[0]),int(value.shape[1])))
  return cache[key]
 M._xsg_rows=xrows
 def xcols(code,value,index):return arr((int(value.shape[0]),int(index.shape[0])))
 M._xsg_cols=xcols;M._W5_PORT_SCOPE=None
 ctx=dict(eligible=True,p=p0,previous=prev,newborn=newborn,young_rows=832,total=896,basis=bq,U=None,rows=rr,tail=tt,w=wf,loss=0,key=(896,832,896,9504),price=(0,0))
 def old():
  M._O18_PEEL_STATE['left']=450
  return M._shared_old384_mm(A,bq),M._O18_FUSION_PARENT_NEWBORN(w,p0,prev,newborn)
 def new():
  ctx['U']=None;M._O18_FUSION_CONTEXT[0]=ctx
  return M._astra_bcc_newborn_impl(w,p0,prev,newborn)
 measure(old);o=measure(old);measure(new);v=measure(new)
 rows.append(dict(rank=rank,base=o,new=v,gain_flops=o['flops']-v['flops'],saved_requests=o['requests']-v['requests']))
report=dict(status='LITERAL_NEWBORN_WRAPPER_WITH_SHARED_GATHERS_STUBBED',rows=rows,models=0,servers=0,numeric_matmuls=0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['Shared row gathers are metadata stubs; extra tail gather costs excluded ~3calls/64*k.','Warm public tariff; owned-result copies and full newborn reconstruction included.'])
assert report['peak_rss_bytes']<500_000_000
with (HERE/'FUSION_LITERAL_PRICE_v1.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
for row in rows:print(json.dumps({k:v for k,v in row.items()if k not in('base','new')}))
