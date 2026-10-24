"""Four deterministic previous-grid peel patterns, current Sm40 L6 body and public _mm tails."""
from pathlib import Path
import gc,json,math,time,resource,sys
D=Path(__file__).resolve().parent;R=D.parents[1];OLD=R/'astra_crunch_compute_v1/bini_apa_v1';t0=time.monotonic()
src=OLD/'price_outer_232_v3.py';ns={'__file__':str(src),'__name__':'peel_metadata'};exec(src.read_text().split('def make_bini')[0],ns)
M,fn,fs,arena,measure,arr=(ns[k]for k in ('M','fn','fs','arena','measure','arr'))
def body(a,b,record):
 m,k,n=int(a.shape[0]),int(a.shape[1]),int(b.shape[1]);L=6;u=64;rec=M._SMX_RECORDS[record];ar,ak,an=rec['shape'];cm,ck,cn=math.ceil(m/(ar*u))*u,math.ceil(k/(ak*u))*u,math.ceil(n/(an*u))*u;r,q,c=cm//u,ck//u,cn//u
 P=next((v for v in (40,20,10,8,5,4,2,1) if 7**L*v*(r*q+q*c+r*c)<=391602176),None)
 if P is None:return None
 fam=('peel',L,P,r,q,c);env=M._SmxEnvironment();env.fnp=fn;env._IPT_REGION=M._IPT_REGION;env._BH_HOLD=M._BH_HOLD;env._TR_LAST=M._TR_LAST;env._SCRATCH=M._SCRATCH;env._rs_nat_perm=M._rs_nat_perm;env._rs_basis_views=M._rs_basis_views
 env._sm40_owner=arena;env._sm40_cache={};env._sm40_io_cache={};env._sm40_outer_cache={};env._sm40_group_cache={};env._sm40_padding_grid={};env._sm40_padding_cache={};env._sm40_extra_leaf_bytes=0
 env._sm40_frames={'peel':M._astra_sm40_bind_family(fn,env,arena,fam)};env._sm40_outer={'A':arr((40,cm,ck)),'B':arr((40,ck,cn)),'C':arr((40,cm,cn))};env._sm40_padding=(arr((ar*cm,ak*ck)),arr((ak*ck,an*cn)))
 ker=M._SmxKernel(env,rec);ker.family=fam
 return lambda:ker(a,b,L,P),dict(record=record,L=L,P=P,body_shape=[m,k,n],child=[cm,ck,cn],arena_elements=7**L*P*(r*q+q*c+r*c))
def make(a,b,record,cuts):
 m,k,n=int(a.shape[0]),int(a.shape[1]),int(b.shape[1]);m0,k0,n0=cuts
 built=body(a[:m0,:k0],b[:k0,:n0],record)
 if built is None:return None
 call,info=built;assembled=m0<m or n0<n;out=arr((m,n)) if assembled else None
 d00=out[:m0,:n0] if assembled else None
 tails=[]
 if k0<k:tails.append(('inner',a[:m0,k0:],b[k0:,:n0]))
 if n0<n:tails.append(('columns',a[:m0,:],b[:,n0:]))
 if m0<m:tails.append(('rows',a[m0:,:],b))
 dests={'columns':out[:m0,n0:] if n0<n else None,'rows':out[m0:,:] if m0<m else None}
 def run():
  main=call();answer=out if assembled else main
  if assembled:fn.copyto(d00,main)
  for kind,A,B in tails:
   z=M._mm(A,B)
   if kind=='inner':fn.add(d00 if assembled else answer,z,out=d00 if assembled else answer)
   else:fn.copyto(dests[kind],z)
  return answer
 info.update(cuts=cuts,tails=[dict(kind=kind,shape=[int(A.shape[0]),int(A.shape[1]),int(B.shape[1])])for kind,A,B in tails],assemble_copy_elements=m*n if assembled else 0,inner_add_elements=m0*n0 if k0<k else 0)
 return run,info

def schemes(m,k,n,record):
 grid=[64*x for x in M._SMX_RECORDS[record]['shape']];dims=[m,k,n];floor=[x//g*g for x,g in zip(dims,grid)];out=[]
 for label,axes in [('m',(0,)),('k',(1,)),('n',(2,)),('all',(0,1,2))]:
  cuts=[floor[i]if i in axes else dims[i]for i in range(3)]
  if min(cuts)>0 and cuts!=dims and all(cuts!=x[1] for x in out):out.append((label,cuts))
 return out

def price_one(spec,kind):
 shape=spec['shape'];m,k,n=shape[:3];rank=shape[3]if kind=='read' else 0;record='signed_executable_1'if kind=='read'else'signed_executable_0';totalK=k+rank;base=spec['base']if kind=='read'else spec['pushsm'];trials=[]
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  A=arr((m,totalK));B=arr((totalK,n));youngA=arr((m,k));youngB=arr((k,n));lam=arr((k,));oldA=arr((m,rank));oldB=arr((rank,n));ya=A[:,:k];yb=B[:k,:];oa=A[:,k:];ob=B[k:,:]
 for label,cuts in schemes(m,totalK,n,record):
  with fs.BudgetContext(flop_budget=2**60,quiet=True):built=make(A,B,record,cuts)
  if built is None:continue
  f,info=built
  def call():
   if kind=='read':
    if m<n:fn.multiply(youngA,lam,out=ya);fn.copyto(yb,youngB)
    else:fn.copyto(ya,youngA);fn.multiply(youngB,lam[:,None],out=yb)
    if rank:fn.copyto(oa,oldA);fn.copyto(ob,oldB)
   return f()
  measure(call);v=measure(call);trials.append(dict(scheme=label,**info,**v,saving=base['flops']-v['flops'],request_delta=v['requests']-base['requests']))
  del call,f,built;M._BH_HOLD.clear();gc.collect()
 best=min(trials,key=lambda x:x['flops'])if trials else None
 return dict(kind=kind,shape=shape,base=base,trials=trials,best=best)
push=json.loads((OLD/'OUTER_COMBINED_TARIFF_v3.json').read_text());read=json.loads((OLD/'READ_TARIFF_v2.json').read_text())
if len(sys.argv)>1 and sys.argv[1]=='example':
 spec=next(x for x in push['rows']if x['shape']==[1024,1024,6144]);r=price_one(spec,'push');(D/'EXAMPLE_v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(0)
rows=[]
for kind,collection in [('push',push),('read',read)]:
 for i,spec in enumerate(collection['rows']):
  row=price_one(spec,kind);rows.append(row)
  if i%20==0:print(json.dumps(dict(kind=kind,progress=i,total=len(collection['rows']),shape=row['shape'],best_saving=row['best']['saving']if row['best']else None)),flush=True)
  (D/'PARTIAL_v1.json').write_text(json.dumps(rows)+'\n')
by={(x['kind'],tuple(x['shape'])):x for x in rows};coverage={}
for kind,col in [('push',push),('read',read)]:
 cov=[]
 for c in col['coverage']:
  eligible=[by[kind,tuple(h)]for h in c['hits']if by[kind,tuple(h)]['best']and by[kind,tuple(h)]['best']['saving']>0];gain=sum(x['best']['saving']for x in eligible);den=c['total']if kind=='push'else c['bill'];cov.append(dict(seed=c['seed'],denominator=den,gain=gain,pct=100*gain/den,request_delta=sum(x['best']['request_delta']for x in eligible),calls=len(eligible)))
 coverage[kind]=cov
rc={x['seed']:x for x in coverage['read']};matched=[dict(seed=x['seed'],denominator=x['denominator'],push_gain=x['gain'],read_gain=rc[x['seed']]['gain'],combined_pct=100*(x['gain']+rc[x['seed']]['gain'])/x['denominator'])for x in coverage['push']]
summary={kind:dict(mean_pct=sum(x['pct']for x in cov)/len(cov),mean_request_delta=sum(x['request_delta']for x in cov)/len(cov),winning_shapes=sum(x['kind']==kind and x['best']is not None and x['best']['saving']>0 for x in rows))for kind,cov in coverage.items()};summary['matched8_combined_mean_pct']=sum(x['combined_pct']for x in matched)/len(matched)
out=dict(status='WARM_PUBLIC_METADATA_PRICE_NO_NUMERIC_OR_RUNTIME_QUALIFICATION',source_sha256=ns['p'].SOURCE_SHA,summary=summary,rows=rows,coverage=coverage,matched8=matched,seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,payload_bytes=ns['p'].PAYLOAD[0],limitations=['Four deterministic peel schemes only: previous L6 grid on m,k,n individually or all.','Body uses current exact Sm40 circuits and L6 actual alternative-basis children; tails current _mm. All body padding, products, assembly, add/copies charged.','Read fuses oldrank384 into contracted axis exactly, with full input packing and lambda multiplication explicitly charged. Unchanged old-factor construction excluded fromboth sides.','No native numeric, setup prebinding, outputalias or runtime qualification.'])
(D/'TARIFF_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ('summary','seconds','maxrss','payload_bytes')}),flush=True)
