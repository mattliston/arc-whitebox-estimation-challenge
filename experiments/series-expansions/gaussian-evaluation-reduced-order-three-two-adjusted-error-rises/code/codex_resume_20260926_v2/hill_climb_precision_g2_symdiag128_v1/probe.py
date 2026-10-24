"""Root-only tiny native Gaussian-term primitive; no estimator/setup/network call.
CLI: NATIVE -B probe.py EXCLUSIVE_RESULT_JSON; guard45s/7.5GB/zero swap.
Compiles two exact source helpers plus selected unchanged Gaussian statements.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D=Path(__file__).resolve().parent
NATIVE=D.parent.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
CANDIDATE_SHA='22a63fa238f04f45a78a18750d8868fcee57a41e212775eac250bf744bf6781f'
PARENT_SHA='88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
BUILD_SHA='674d45899ce2eece3ab0ca9dc89530c927777161edda992eac202463a12daaf7'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
STATEMENTS=(
 "P16 = st['P16']",
 "_e2_hold.append(fnp.divide(rho, st['DIV6'], out=P16))",
 "_e2_hold.append(fnp.cumprod(P16, axis=0, out=P16))",
 "Z0, Z1, Z2, Z3, Z4 = st['Zv']",
 "_e2_A = _e2_pair_arena(n)",
 "X = _e2_A['X1']",
 "_e2_hold.append(fnp.multiply(st['Fr16'], st['Gr16'], out=X))",
 "_e2_hold.append(fnp.multiply(X, st['P6'], out=X))",
 "_e2_hold.append(fnp.sum(X, axis=0, out=Z0))",
)


def pin(path):
 p=Path(path).resolve();r=p.read_bytes()
 return dict(path=str(p),sha256=hashlib.sha256(r).hexdigest(),bytes=len(r))


def bill(ctx):
 s=ctx.summary_dict()
 return dict(flops=int(s['flops_used']),operations=sum(int(v.get('calls',v.get('count',0))) for v in s['operations'].values()))


def extracted(path):
 tree=ast.parse(path.read_text());functions={}
 for name in ('_e2_pair_state','_e2_pair_arena','_nonlinear_pair_batch'):
  found=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
  assert len(found)==1,name
  functions[name]=found[0]
 picked=[]
 for statement in STATEMENTS:
  target=ast.dump(ast.parse(statement).body[0],include_attributes=False)
  found=[n for n in functions['_nonlinear_pair_batch'].body if ast.dump(n,include_attributes=False)==target]
  assert len(found)==1,statement
  picked.append(copy.deepcopy(found[0]))
 function=ast.parse('def gaussian(st, rho, n):\n return Z0\n').body[0]
 function.body=picked+function.body
 module=ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(functions['_e2_pair_state']),copy.deepcopy(functions['_e2_pair_arena']),function],type_ignores=[]))
 namespace=dict(fnp=fnp,_E2_PAIR_STATE={},_E2_PAIR_ARENA={},_e2_hold=[])
 exec(compile(module,str(path),'exec',dont_inherit=True),namespace)
 return namespace


def main():
 assert len(sys.argv)==2 and Path(sys.executable).absolute()==NATIVE and np.__version__=='2.4.6'
 assert all(os.environ.get(k)=='1' for k in THREADS)
 target=Path(sys.argv[1]).resolve();assert not target.exists()
 bp=pin(D/'build.json');assert bp['sha256']==BUILD_SHA
 build=json.loads((D/'build.json').read_text())
 for key in ('candidate','parent','builder','protocol','pricing'):assert pin(build[key]['path'])==build[key]
 assert build['source_inverse_exact'] and build['candidate']['sha256']==CANDIDATE_SHA and build['parent']['sha256']==PARENT_SHA
 source=Path(build['candidate']['path']).read_text();inverse=source
 for edit in reversed(build['edits']):
  assert inverse.count(edit['after'])==1
  inverse=inverse.replace(edit['after'],edit['before'],1)
 assert inverse==Path(build['parent']['path']).read_text()
 n=8;nn=n*n
 F=((np.arange(7*3*n).reshape(7,3,n)%7)-3).astype(np.float32)/4
 G=((np.arange(7*3*n).reshape(7,3,n)%5)-2).astype(np.float32)/4
 records=[]
 for label,R in (('parent',3),('candidate',2)):
  ns=extracted(Path(build[label]['path']))
  with flops.BudgetContext(flop_budget=2**30,quiet=True) as ctx:
   ns['_SCRATCH']=(fnp.empty(72*nn,dtype=fnp.float32),)
   st=ns['_e2_pair_state'](n);views=ns['_e2_pair_arena'](n)
  binding=bill(ctx)
  assert tuple(st['P16'].shape)==(R,n,n) and tuple(views['X1'].shape)==(R,3,n,n)
  assert tuple(st['RR'].shape)==(3,1,n,n) and tuple(st['P12'].shape)==(2,1,n,n)
  with flops.BudgetContext(flop_budget=2**30,quiet=True):
   fnp.copyto(st['FT'],F);fnp.copyto(st['GT'],G)
  cases=[]
  for shift in (0,2):
   rho=(((np.arange(nn).reshape(n,n)+shift)%9)-4).astype(np.float32)/8
   powers=np.cumprod(rho[None]/np.arange(1,R+1,dtype=np.float32)[:,None,None],axis=0)
   reference=((F[1:R+1,:,:,None]*G[1:R+1,:,None,:])*powers[:,None]).sum(axis=0)
   hashes=[];costs=[]
   for repeat in range(2):
    with flops.BudgetContext(flop_budget=2**30,quiet=True):
     rr=fnp.asarray(rho);fnp.copyto(st['P16'],np.float32(np.nan))
    ns['_e2_hold'].clear()
    with flops.BudgetContext(flop_budget=2**30,quiet=True) as ctx:
     out=ns['gaussian'](st,rr,n)
    result=np.asarray(out).copy();costs.append(bill(ctx))
    assert result.dtype==np.float32 and np.isfinite(result).all()
    assert np.array_equal(result,reference)
    assert np.array_equal(np.asarray(st['P12'])[:,0],powers[:2])
    assert np.array_equal(np.asarray(st['RR'])[0,0],np.ones((n,n),dtype=np.float32))
    assert np.array_equal(np.asarray(rr),rho) and np.array_equal(np.asarray(st['FT']),F) and np.array_equal(np.asarray(st['GT']),G)
    hashes.append(hashlib.sha256(result.tobytes()).hexdigest())
   assert hashes[0]==hashes[1] and costs[0]==costs[1]
   cases.append(dict(shift=shift,reference_bytes_exact=True,repeat_bytes_exact=True,powers_fresh=True,inputs_unchanged=True,output_sha256=hashes[0],**costs[0]))
  records.append(dict(role=label,R=R,binding=binding,cases=cases))
  ns.clear()
 parent,child=records
 assert parent['binding']['flops']-child['binding']['flops']==3*nn+2
 for p,c in zip(parent['cases'],child['cases']):
  assert p['flops']-c['flops']==11*nn
  assert p['operations']==c['operations']
 for key in ('candidate','parent','builder','protocol','pricing'):assert pin(build[key]['path'])==build[key]
 assert pin(D/'build.json')==bp
 result=dict(passed=True,source=build['candidate'],parent=build['parent'],build=bp,probe=pin(__file__),cases=records,
  warm_flop_saving_per_case=11*nn,binding_flop_saving=3*nn+2,operation_count_unchanged=True,
  scope='Exact source helpers and Gaussian statements only; E2/EL2 unchanged by inverse source proof, endpoint accuracy remains a native pilot gate.',
  model_executed=False,network_targets_read=False,serving_qualified=False,external_guard_required=True)
 with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('GAUSSIAN_R2_SOURCE_PRIMITIVE_COMPLETE',flush=True)


if __name__=='__main__':main()
