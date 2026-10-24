"""Standalone native/RPC SLR predicate, eager failure and exact tariff proof."""
import argparse,ast,hashlib,importlib.util,json,math,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['native','rpc'],required=True);a=p.parse_args()
if a.mode=='rpc':assert importlib.util.find_spec('numpy') is None
import flopscope as fs
from flopscope import numpy as fnp
HERE=Path(__file__).resolve().parent;source=HERE.parent/'runtime_first128_dcr_scatter_v1/candidate_first128_smoff_npxoff_dcrscatter_release_v1.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(source)=='90f01bb9893dd8aaa900ccc5c9a7ade4999c3f41ef2da4575f6b2dc81085be5c';assert fs.__version__.startswith('0.12.3')
E=dict(fnp=fnp,_E2Hold=list);tree=ast.parse(source.read_text())
for name in ('_slr_require','_slr_virtual_rotation_envelope'):
 n=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name];assert len(n)==1
 exec(compile(ast.Module(body=n,type_ignores=[]),str(source),'exec'),E)
parent=E['_slr_virtual_rotation_envelope'];exec((HERE/'tail_slr_cholesky_v1.py').read_text(),E);candidate=E['_slr_virtual_rotation_envelope'];assert E['_O21_SLR_CHOLESKY_PARENT'] is parent
assert not {'bool','all','float','int','item','tolist'}.intersection(candidate.__code__.co_names)
records=[];L=float(2**80)
cases=[('zero',0.,0.,True),('signed_zero',-0.,-0.,True),('unit',1.,1.,True),('map_limit',L,0.,True),('base_limit',0.,L,True),('map_limit_small',L,2**-10,True),('base_limit_small',2**-10,L,True),('negative_limit',-L,2**-10,True),('max_product',2**40,2**30,True),('product_boundary',1.,L/384,True),('below_product',1.,math.nextafter(L/384,0.),True),('above_product_rounds_to_boundary',1.,math.nextafter(L/384,math.inf),True),('above_product',1.,math.nextafter(math.nextafter(L/384,math.inf),math.inf),False),('large_product',2**40,2**40,False),('above_map',math.nextafter(L,math.inf),0.,False),('above_base',0.,math.nextafter(L,math.inf),False),('inf_map',math.inf,0.,False),('inf_base',0.,math.inf,False),('nan_map',math.nan,1.,False),('nan_base',1.,math.nan,False),('both_nan',math.nan,math.nan,False),('negative_inf',-math.inf,1.,False),('max_fp32',float.fromhex('0x1.fffffep127'),0.,False),('smallest_fp32',2**-149,2**-149,True)]
for name,x,y,expected in cases:
 # FP64 also covers exact nextafter predicate boundaries; production inputs
 # are FP32 and use the same subsequent FP64 maxima/predicate operations.
 for dtype_name in ('float64','float32'):
  if dtype_name=='float32' and name in ('above_map','above_base','above_product','above_product_rounds_to_boundary','below_product'):continue
  observed=[]
  for label,fun in [('parent',parent),('candidate',candidate)]:
   with fs.BudgetContext(flop_budget=10**7,quiet=True) as ctx:
    overlap=fnp.full((2,3),x,dtype=getattr(fnp,dtype_name));base=fnp.full((3,4),y,dtype=getattr(fnp,dtype_name))
    before=ctx.flops_used;returned=False;exception=None
    try:fun(overlap,base);returned=True
    except Exception as exc:exception=dict(type=type(exc).__name__,message=str(exc))
    used=ctx.flops_used-before
    # Record before any fetch, any following backend op, or context exit.
    observed.append(dict(label=label,returned_before_next_op=returned,exception=exception,flops=used))
    if not returned:assert exception==dict(type='RuntimeError',message='Shared lazy rotation: virtual rotation overflow envelope'),(name,dtype_name,label,exception)
   # No output is fetched: successful factor result is intentionally unused.
  assert observed[0]['returned_before_next_op']==observed[1]['returned_before_next_op'],(name,dtype_name,observed)
  if dtype_name=='float64':assert observed[0]['returned_before_next_op']==expected,(name,expected,observed)
  records.append(dict(case=name,dtype=dtype_name,expected_exact_float64=expected,parent=observed[0],candidate=observed[1],same_acceptance_and_exception=True,flop_delta=observed[1]['flops']-observed[0]['flops']))
# Direct primitive eager failure test, independently showing no later fetch
# or use of the Cholesky result is needed to trigger its backend evaluation.
primitive=[]
for flag in (True,False):
 with fs.BudgetContext(flop_budget=1000,quiet=True) as ctx:
  allowed=fnp.full((1,1),flag,dtype=fnp.bool_);before=ctx.flops_used;returned=False;error=None
  try:fnp.linalg.cholesky(allowed.astype(fnp.float64));returned=True
  except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
  used=ctx.flops_used-before
  assert returned==flag,(flag,returned,error)
  primitive.append(dict(flag=flag,returned_before_next_op=returned,error=error,flops=used))
report=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),successful=True,mode=a.mode,official_version=fs.__version__,source_sha256=sha(source),tail_sha256=sha(HERE/'tail_slr_cholesky_v1.py'),script_sha256=sha(__file__),records=records,primitive=primitive,no_array_materialization_in_candidate=True,check_executes_even_when_result_unused=True,all_original_predicate_arithmetic_preserved=True,model_count=0,truth_count=0,runtime_benefit_unmeasured=True)
with(HERE/('SLR_'+a.mode.upper()+'_CHECK_v2.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(mode=a.mode,successful=True,cases=len(records),valid_debits=sorted(set(r['flop_delta'] for r in records if r['parent']['returned_before_next_op'])),primitive=primitive)))
