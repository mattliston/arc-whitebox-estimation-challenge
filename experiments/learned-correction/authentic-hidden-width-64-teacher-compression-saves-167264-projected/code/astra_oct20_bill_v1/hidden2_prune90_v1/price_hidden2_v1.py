"""Complete official GNN-call tariff change for the one frozen hidden64 shape.
No capture arrays, truth, fitted parameters, estimator setup or predict.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import ast,base64,hashlib,json,subprocess,sys,zlib
from pathlib import Path
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp
HERE=Path(__file__).resolve().parent;V=HERE.parents[1]/'astra_oct20_vm_frontier_v1';RUNS=Path('/home/arcworker/arc-runs')
sys.path.insert(0,str(V));import fit_frontier_v1 as f;import check_frontier_tiny_v6 as tiny
assert f.sha(tiny.__file__)=='a6246110ed4e3aaeba9f49b51b9bb3bd855e6656fc3508e8703b6c1c45aac376'
assert f.sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
assert f.sha(HERE/'PREREG_v1.json')=='97c868a0a6ee0a12850c97af7a041f049d929b0c950df920d65fbf246425cb86'
assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_SERVER_URL') and not os.environ.get('FLOPSCOPE_WEIGHTS_FILE')
source=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510';assert f.sha(source)==SCI
text=source.read_text();tree=ast.parse(text);literal,_=f.literals_and_functions(text);P=f.decode(literal['_GNN_LITERAL']);code=tiny.load_functions(text)
E=ast.literal_eval([n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_VMFR_ORIGINAL_W3_ENCODER' for t in n.targets)][-1])
Q={k:v.copy() for k,v in P.items()};Q['enc2_w']=Q['enc2_w'][:64].copy();Q['enc2_b']=Q['enc2_b'][:64].copy();Q['enc4_w']=Q['enc4_w'][:,:64].copy()
assert P['enc2_w'].shape==(128,128) and Q['enc2_w'].shape==(64,128) and Q['enc4_w'].shape==(64,64)
def env():
 d=dict(fnp=fnp,_static_base64=base64,_static_zlib=zlib,_GOP2_ON=True,_GOP2_ACTIVE=[True],_GNN_PARAMS=None,
  _E2_GNN_STATE={},_E2_MSG={},_FAST_RESCUE_FRAC={},_HC_CONST={},_HCF_CONST={},_MESSAGE_EDGE_PARAMETERS=None,
  _GOP2_STATS=dict(calls=0,layers=[],energy_blocks=0,message_misses=0),_W3R_STATS=dict(calls=0,metric_builds=0),
  _VMFR_CM=8,_VMFR_ORIGINAL_W3_ENCODER=E,_VMFR_W3_ENCODER_ARRAY=None,_W3R_METRIC=None)
 exec(code,d);return d
rows=[];rng=np.random.default_rng(202610210328)
for n in (16,1024):
 W=rng.normal(0,np.sqrt(2/n),(n,n)).astype('f4');state=rng.normal(0,.3,(n,64)).astype('f4');sigma=rng.uniform(.2,1.2,n).astype('f4')
 inputs={l:(rng.normal(0,.5,(n,30))*P['scales'][l]+P['centers'][l]).astype('f4') for l in range(1,16)}
 bias=(P['enc0_b']+rng.normal(0,.03,(n,128))).astype('f4');costs=[]
 for label,params in [('parent128',P),('student64',Q)]:
  d=env();records=[]
  with fs.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
   p={k:fnp.asarray(v) for k,v in params.items()};d['_GNN_PARAMS']=p
   sig,weights,st,bi=[fnp.asarray(v) for v in (sigma,W,state,bias)];xx={l:fnp.asarray(v) for l,v in inputs.items()}
   w2=weights*weights;w3=w2*weights;w4=w3*weights;w6=w3*w3
   wp=dict(T013=fnp.stack((weights.T,w2.T,w3.T)),T135=fnp.stack((w2.T,w4.T,w6.T)))
   d['_e2_wpow']=lambda W,need6=False:wp;d['_hc_bias']=lambda l,unused:bi
   saved={}
   for phase in ('cold','warm'):
    for layer in range(1,16):
     d['_E2_MSG'].clear()
     if 8<=layer<=14:d['_e2_messages'](d['_e2_gnn_consts'](p,n),weights,st)
     before,count=ctx.flops_used,len(ctx.op_log);delta,got=d['_gop2_gnn_step'](layer,xx[layer],sig,weights,st,p)
     bill,calls=ctx.flops_used-before,len(ctx.op_log)-count
     values=(np.asarray(got).copy(),np.asarray(delta).copy());assert np.isfinite(values[0]).all() and np.isfinite(values[1]).all()
     if phase=='cold':
      saved[layer]=values
      if n==16:
       expected,edelta=tiny.oracle(params,np.asarray(E,'f8'),layer,inputs[layer],sigma,W,state,bias)
       np.testing.assert_allclose(values[0],expected,rtol=5e-5,atol=5e-5);np.testing.assert_allclose(values[1],edelta,rtol=1e-4,atol=2e-8)
     else:
      np.testing.assert_array_equal(saved[layer][0],values[0]);np.testing.assert_array_equal(saved[layer][1],values[1])
     records.append(dict(layer=layer,phase=phase,flops=int(bill),calls=calls))
   np.testing.assert_array_equal(np.asarray(st),state);np.testing.assert_array_equal(np.asarray(weights),W)
  costs.append(dict(model=label,records=records))
  print(json.dumps(dict(utc=f.utc(),n=n,model=label,complete=True)),flush=True)
 for phase in ('cold','warm'):
  old=[r for r in costs[0]['records'] if r['phase']==phase];new=[r for r in costs[1]['records'] if r['phase']==phase]
  delta=sum(r['flops'] for r in new)-sum(r['flops'] for r in old);op_delta=sum(r['calls'] for r in new)-sum(r['calls'] for r in old)
  assert delta<0 and op_delta==0
  rows.append(dict(n=n,phase=phase,parent_full_gnn_flops=sum(r['flops'] for r in old),student_full_gnn_flops=sum(r['flops'] for r in new),network_flop_change=delta,calls_change=op_delta,parent=old,student=new))
control=RUNS/'msg8_loaderfix337422_control32_markline_blas4_v1';manifest=json.loads((control/'RUN_MANIFEST.json').read_text());assert manifest['candidate_sha256']==SCI
den=[]
for p in sorted(control.glob('part*/rows.jsonl')):
 for line in p.read_text().splitlines():
  r=json.loads(line);assert r['phase']=='scored';den.append(dict(rank=r['rank'],flops=r['flops']))
assert len(den)==32 and len({r['rank'] for r in den})==32
price={r['phase']:dict(flops=r['network_flop_change'],mean_projected_bill_pct=float(np.mean([100*r['network_flop_change']/x['flops'] for x in den]))) for r in rows if r['n']==1024}
assert price['cold']['flops']==price['warm']['flops']
out=RUNS/'hidden2_prune90_price_v1';out.mkdir(exist_ok=False)
result=dict(utc=f.utc(),successful=True,status='COMPLETE_GNN_TARIFF_MEASURED_NO_ACCURACY_CLAIM',source_sha256=SCI,script_sha256=f.sha(__file__),prereg_sha256=f.sha(HERE/'PREREG_v1.json'),
 official_version=fs.__version__,rows=rows,projected_network_bill=price,reference_submission=337422,reference_control_manifest_sha256=f.sha(control/'RUN_MANIFEST.json'),
 passes_price_gate=price['warm']['mean_projected_bill_pct']<=-.10,all15_gnn_calls_complete=True,unchanged_message_precompute_excluded_identically=True,
 parameters_put_outside_measured_calls=True,source_functions_unchanged=True,no_captures_or_truth_read=True,no_adjusted_gain_claim=True,no_candidate_export=True)
f.jwrite(out/'PRICE_v1.json',result);print(json.dumps(dict(status=result['status'],price=price,passes_price_gate=result['passes_price_gate'])),flush=True)
