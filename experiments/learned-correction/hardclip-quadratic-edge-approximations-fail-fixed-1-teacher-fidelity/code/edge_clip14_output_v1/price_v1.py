"""Complete-edge official tiny tariff, then existing first7 roster projection.
No estimator setup/predict, no truth or fit arrays; source remains unchanged.
"""
import ast, hashlib, json, resource, subprocess, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
V=HERE.parent
R=V.parent
SOURCE=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
SOURCE_SHA='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
OFFICIAL=R/'astra_crunch_runtime_v1/failure336631_v1/flopscope_0.12.3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def main():
 assert sha(SOURCE)==SOURCE_SHA
 out=HERE/'PRICE_v1.json'; assert not out.exists()
 metadata=json.loads((HERE/'CAPTURE_PRICE_INPUT_v1.json').read_text())
 assert metadata['source_sha256']==SOURCE_SHA and metadata['reference_sha256']==SOURCE_SHA
 assert sorted(metadata['rows'])==['150','200','250','259','277','287','35']
 sys.path.insert(0,str(OFFICIAL))
 import flopscope as fs
 from flopscope import numpy as fnp
 assert fs.__version__.startswith('0.12.3+')
 tree=ast.parse(SOURCE.read_text())
 body={name:[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name][-1]
       for name in ('_ecut_raw','_E2Hold')}
 oldtext=ast.unparse(body['_ecut_raw'])
 lines=oldtext.splitlines()
 remove=("w = fnp.empty(","hold.append(fnp.multiply(h, h, out=w))",
         "hold.append(fnp.add(w, _O20_ECUT_RADICAL_B, out=w))",
         "hold.append(fnp.sqrt(w, out=w))",
         "hold.append(fnp.add(w, _O20_ECUT_RADICAL_A, out=w))",
         "hold.append(fnp.divide(h, w, out=h))")
 killed=[s for s in lines if any(s.strip().startswith(x) for x in remove)]
 assert len(killed)==6,killed
 newtext='\n'.join(s for s in lines if s not in killed)
 env=dict(fnp=fnp,_O20_ECUT_RADICAL_ON=True,_ECUT_STANDIN=True,
          _O20_ECUT_RADICAL_A=-.2684473340048872,_O20_ECUT_RADICAL_B=1.5215732663431163,
          _ECUT_STATS=dict(layers=0,kept=0,full=0))
 exec(compile(ast.Module(body=[body['_E2Hold']],type_ignores=[]),'<hold>','exec'),env)
 exec(compile(oldtext,'<exact777a_edge>','exec'),env); old=env['_ecut_raw']
 exec(compile(newtext,'<clip14_only>','exec'),env); new=env['_ecut_raw']
 rng=np.random.default_rng(20261021); cases=[]
 with fs.BudgetContext(flop_budget=2**35,quiet=True) as budget:
  for n,ni,nj in [(3,1,2),(8,3,5),(16,16,16)]:
   def arr(shape,scale=1):return fnp.asarray((rng.normal(size=shape)*scale).astype('f4'))
   left,right=arr((n,14)),arr((n,14))
   t=dict(bias=arr((14,),.1),output=arr((14,),.1))
   k=dict(R=arr((n,n),.1),S=arr((n,n),.1),e0=arr((14,),.1),e1=arr((14,),.1),e2=arr((14,),.1))
   pf=fnp.asarray(rng.uniform(.1,.4,n).astype('f4')); gf=fnp.asarray(rng.uniform(.1,1,n).astype('f4'))
   env['_E2_LAYER']=dict(ecut_rows=fnp.arange(ni,dtype=fnp.int64),ecut_cols=fnp.arange(nj,dtype=fnp.int64))
   x,c0=budget.flops_used,len(budget.op_log); y0=np.asarray(old(left,right,t,k,pf,gf,[]))
   oldf,oldc=budget.flops_used-x,len(budget.op_log)-c0
   x,c0=budget.flops_used,len(budget.op_log); y1=np.asarray(new(left,right,t,k,pf,gf,[]))
   newf,newc=budget.flops_used-x,len(budget.op_log)-c0
   assert oldf-newf==5*ni*nj*14 and oldc-newc==6, (n,ni,nj,oldf,newf,oldc,newc)
   assert np.isfinite(y0).all() and np.isfinite(y1).all()
   cases.append(dict(n=n,rows=ni,cols=nj,hidden=14,old_flops=oldf,new_flops=newf,
                     old_calls=oldc,new_calls=newc,credit_flops=oldf-newf))
 rows=[]
 for rank,r in metadata['rows'].items():
  assert sorted(x['layer'] for x in r['layers'])==list(range(8,15))
  elements=sum(x['rows']*x['cols']*14 for x in r['layers'])
  rows.append(dict(rank=int(rank),elements=elements,credit_flops=elements*5,
                   current_bank_flops=r['reference_flops'],bill_pct=-500*elements/r['reference_flops']))
 report=dict(utc=utc(),status='COMPLETE_RAW_EDGE_TARIFF_PASS_ROSTER_PROJECTION_ONLY',
      source_sha256=SOURCE_SHA,reference_submission=337422,reference_sha256=SOURCE_SHA,
      activation='clip(z,-1,1); no slope/offset fitting',width=14,
      cases=cases,first7=rows,first7_mean_bill_pct=float(np.mean([r['bill_pct'] for r in rows])),
      warm_calls_saved_per_network=42,cold_literal_shape_unchanged=True,
      metadata_sha256=sha(HERE/'CAPTURE_PRICE_INPUT_v1.json'),script_sha256=sha(__file__),
      sdk=fs.__version__,tariff_sha256=sha(OFFICIAL/'flopscope/data/default_weights.json'),
      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      limitations=['Seven existing teacher roster shapes; downstream candidate roster drift is not priced.',
                  'No output fit, native prediction, MSE or adjusted gain measured.'])
 with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()

