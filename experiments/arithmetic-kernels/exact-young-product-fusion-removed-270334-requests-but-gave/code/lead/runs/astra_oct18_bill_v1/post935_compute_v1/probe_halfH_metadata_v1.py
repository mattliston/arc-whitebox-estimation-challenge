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

p=ns['p']; np=p.np; oldbackend=p.backend
def reductions_backend(fun,*args,**kwargs):
 name=getattr(fun,'__name__','')
 if name=='sum':
  p.CALLS[name]+=1;a=args[0];axis=kwargs.get('axis');keep=kwargs.get('keepdims',False)
  axes=tuple(range(a.ndim))if axis is None else ((axis,)if isinstance(axis,int)else tuple(axis))
  axes=tuple(x%a.ndim for x in axes);sh=tuple(1 if i in axes else d for i,d in enumerate(a.shape))if keep else tuple(d for i,d in enumerate(a.shape)if i not in axes)
  return p.virtual(sh,kwargs.get('dtype')or a.dtype)
 if name in ('argsort','sort'):
  p.CALLS[name]+=1;return p.virtual(args[0].shape,np.int64 if name=='argsort'else args[0].dtype)
 return oldbackend(fun,*args,**kwargs)
for name,mod in list(sys.modules.items()):
 if name.startswith('flopscope')and getattr(mod,'_call_numpy',None)is oldbackend:mod._call_numpy=reductions_backend
helper=R/'astra_oct18_accuracy_v1/post336935_accuracy_v1/fragment_cp_halfH_v1.py'
exec(compile(helper.read_text(),str(helper),'exec'),M.__dict__)
results=[]
for layer,m,width,kc in [(7,960,6144,2048),(10,832,5568,1536),(13,832,4672,1536)]:
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  J=arr((m,width));weight=arr((width,));scores=arr((width,));rows=fn.asarray(list(range(384)),dtype=fn.int64)
 first,last=M._JOIN_SCHEDULE[layer];cohorts=[None]*(last+1);offset=0;left=width
 for birth in range(first,last+1):
  w=left if birth==last else 2*(left//(2*(last-birth+1)))
  cohorts[birth]=(offset,w);offset+=w;left-=w
 state=dict(cohorts=cohorts)
 def baseline():
  inds=fn.argsort(-scores.astype(fn.float64),kind='stable')[:kc]
  cols=J[:,inds]
  return M._shared_old384_mm(cols,weight[inds][:,None]*cols[rows,:].T)
 def candidate():
  cols,weights=M._o18_cph_columns(layer,J,weight,scores,state,0)
  return M._shared_old384_mm(cols,weights[:,None]*cols[rows,:].T)
 measure(baseline);b=measure(baseline);measure(candidate);c=measure(candidate)
 v=dict(layer=layer,m=m,Jwidth=width,kc=kc,base=b,candidate=c,gain_flops=b['flops']-c['flops'],requests_saved=b['requests']-c['requests']);results.append(v)
 print(json.dumps({k:v[k]for k in('layer','gain_flops','requests_saved')}),flush=True)
r=dict(status='ACTUAL0123_WARM_METADATA_NO_EFFICACY',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),sdk=fs.__version__,rows=results,total_gain_flops=sum(x['gain_flops']for x in results),total_requests_saved=sum(x['requests_saved']for x in results),maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,models=0,servers=0,limitations=['One representative actualrank35 joinshape set, not32population claim.','All helperformation/scoring/gathering/casts included; originaldiagrowselection,oldterm,QRandfullFcommon excludedboth.','Only metadata backend; no numericalmatrix execution/efficacy orwall claim.','Cold binding/freehandletraffic notincluded.'])
(D/'HALFH_METADATA_PRICE_v1.json').open('x').write(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='rows'}))
