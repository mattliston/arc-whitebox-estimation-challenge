"""All six full-source imports, exact paid GNN tariff, and one official observer probe.

Synthetic geometry only; no full-network prediction or capture/target arrays.
Diagnostic observer materialization is disclosed, never runtime credit.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import ast, gzip, hashlib, importlib.util, inspect, json, subprocess, sys, tempfile, types
from pathlib import Path
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
PREREG='42c4f80e65348f149e4b64d8e91da29f0fb1995abb5a78f8004fe0623a4a5bd9'
MANIFEST='28a429eac8dc771b6b88aae566abb1423efaf673600eff013ba4bc104ab03397'
PARENT='49bd2ba31f0bce983f44cc001746e63ffaa84e530d3b31efa6424f69a0298c80'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def module(name,path,pin):
 assert sha(path)==pin
 spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out
gp=module('op1_paid_geometry_frozen_helpers',HERE.parent/'first128_gate90_v1/price_first_gate_v1.py','0ab2893e7a99367e2dc76bcd758075d656187dc9646a9379086d0953582c57d5')
f=gp.f
hook=module('op1_source_observer',HERE/'first128_onpolicy_hook_v2.py','f7e2bf4da1ff67ae3249f70accfbc0bd5419135ec951b233ba4a7620dc6ee229')
def rec(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def write(p,x):
 with Path(p).open('x') as stream:json.dump(x,stream,indent=2,allow_nan=False);stream.write('\n')
def oracle():
 node=ast.parse(inspect.getsource(gp.tiny.oracle)).body[0];node.name='oracle_hidden'
 ret=[n for n in node.body if isinstance(n,ast.Return)][-1]
 ret.value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load()) for k in ('out','delta','h','first_hidden')],ctx=ast.Load())
 first=[n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='h' for t in n.targets)][0]
 node.body.insert(node.body.index(first)+1,ast.parse('first_hidden=h.copy()').body[0]);ast.fix_missing_locations(node)
 env=dict(np=np,gop=gp.tiny.gop,w3=gp.tiny.w3)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<independent-first128-oracle>','exec'),env)
 return env['oracle_hidden']

def geometry(family,params,code,constants,fixture,n,terminal_hc,reference,with_observer=False):
 weights,state,sigma,inputs,raw=fixture
 env=gp.environment(code,constants,np.arange(128));records=[];saved={};states={};deltas={};observer=None;capture_proofs=[]
 with tempfile.TemporaryDirectory(prefix='arc_op1_official_observer_') as tmp:
  try:
   with fs.BudgetContext(flop_budget=2**42,quiet=True) as ctx:
    before,count=ctx.flops_used,len(ctx.op_log);p=env['_unpack_static_fp32'](gp.encoded(params));env['_GNN_PARAMS']=p
    setup,setup_calls=ctx.flops_used-before,len(ctx.op_log)-count
    sig,ww,st=[fnp.asarray(v) for v in (sigma,weights,state)]
    xx={layer:fnp.asarray(v) for layer,v in inputs.items()}
    rr={layer:tuple({key:(fnp.asarray(x),None if mask is None else fnp.asarray(mask),count) for key,(x,mask,count) in item.items()} for item in pair) for layer,pair in raw.items()}
    w2=ww*ww;w3=w2*ww;w4=w3*ww;w6=w3*w3
    powers=dict(T013=fnp.stack((ww.T,w2.T,w3.T)),T135=fnp.stack((w2.T,w4.T,w6.T)))
    env['_e2_wpow']=lambda W,need6=False:powers
    if with_observer:
     assert n==1024 and terminal_hc and family['key']=='full90'
     source=ROOT/family['science']['path'];lit,_=f.literals_and_functions(source.read_text())
     mod=types.SimpleNamespace(**env,__file__=str(source),_GNN_LITERAL=lit['_GNN_LITERAL'])
     recipe=dict(teacher=family['science'],target_free=True,uses_network_truth=False,rows_per_layer=1024,layers=list(range(1,16)),
      reference_submission=337466,capture_ranks=[7100120],source_family=family,source_manifest=rec(HERE/'PARENT_SOURCES_v1.json'),
      on_policy_prereg=rec(HERE/'PREREG_v1.json'),original_group_indices=[0])
     observer=hook.install(mod,tmp,recipe)
    for phase in ('cold','warm'):
     capture_phase='warmup' if phase=='cold' else 'scored'
     if observer:observer.begin(7100120,capture_phase)
     for layer in range(1,16):
      env['_E2_MSG'].clear()
      if 8<=layer<=14:env['_e2_messages'](env['_e2_gnn_consts'](p,n),ww,st)
      hc,hcf=rr[layer];env['_HC_STATE'].update(l=layer,raw={} if layer==15 and not terminal_hc else dict(hc));env['_HCF_STATE']['raw']=dict(hcf)
      before,count=ctx.flops_used,len(ctx.op_log)
      delta,got=env['_gop2_gnn_step'](layer,xx[layer],sig,ww,st,p)
      bill,calls=int(ctx.flops_used-before),len(ctx.op_log)-count
      values=np.asarray(got).copy(),np.asarray(delta).copy();assert all(np.isfinite(v).all() for v in values)
      key=phase,layer;states[key]=values[0].tobytes();deltas[key]=values[1].tobytes()
      if reference is not None:
       assert states[key]==reference['states'][key]
       if observer:assert deltas[key]==reference['deltas'][key]
      if phase=='cold':
       saved[layer]=values
       if n==16:
        hc_raw,hcf_raw=raw[layer];bias=gp.standardized_bias(params,constants,np.arange(128),{} if layer==15 and not terminal_hc else hc_raw,hcf_raw,n)
        expected,edelta,h2,h1=ORACLE(params,np.asarray(constants['_VMFR_ORIGINAL_W3_ENCODER'],'f8'),layer,inputs[layer],sigma,weights,state,bias)
        scale=sigma*params['rms'][layer]*(1+.25*np.tanh(params['gain'][layer:layer+1]))
        edelta+=scale*(h2@params['hidden_head'][layer]+h1@params['first_hidden_head'][layer])
        np.testing.assert_allclose(values[0],expected,rtol=5e-5,atol=5e-5);np.testing.assert_allclose(values[1],edelta,rtol=1e-4,atol=2e-8)
      else:
       np.testing.assert_array_equal(values[0],saved[layer][0]);np.testing.assert_array_equal(values[1],saved[layer][1])
      records.append(dict(layer=layer,phase=phase,flops=bill,calls=calls))
      if observer:
       cap=Path(tmp)/'gnn_full_state_v1'/('rank7100120_%s_L%02d.json.gz'%(capture_phase,layer))
       with gzip.open(cap,'rt') as stream:data=json.load(stream)
       hh=np.asarray(data['first128']['data'],'f4');assert data['first128']['shape']==[1024,128]
       H=np.asarray(env['_e2_gnn_consts'](p,n)['H']).copy()
       hc_raw,hcf_raw=raw[layer];bias=gp.standardized_bias(params,constants,np.arange(128),hc_raw,hcf_raw,n)
       pre=H@params['enc0_w'].T+bias;expected=pre/(np.float32(1)+np.exp(-pre))
       err=float(np.max(np.abs(hh-expected)));np.testing.assert_allclose(hh,expected,rtol=3e-4,atol=3e-5)
       np.testing.assert_array_equal(np.asarray(data['state']['data'],'f4'),values[0])
       np.testing.assert_array_equal(np.asarray(data['delta']['data'],'f4'),values[1])
       capture_proofs.append(dict(phase=phase,layer=layer,first128_independent_float32_replay_max_abs=err,
        captured_state_delta_byte_exact=True,capture_sha256=sha(cap)))
    np.testing.assert_array_equal(np.asarray(st),state);np.testing.assert_array_equal(np.asarray(ww),weights)
   if observer:
    completion=observer.finish();assert len(completion['files'])==30 and completion['direct_first128_count']==30
    assert completion['all_first_paired_once'] and not completion['native_handles_or_frames_retained']
  finally:
   if observer:observer.abort('official qualification exit')
 return dict(family=family['key'],source_sha256=family['science']['sha256'],n=n,terminal_hc_present=terminal_hc,
  parameter_decode_flops=int(setup),parameter_decode_calls=setup_calls,records=records,states=states,deltas=deltas,
  observer=with_observer,capture_proofs=capture_proofs)

def main():
 global ORACLE
 assert sys.platform=='linux' and fs.__version__.startswith('0.12.3')
 assert not os.environ.get('FLOPSCOPE_SERVER_URL') and not os.environ.get('FLOPSCOPE_WEIGHTS_FILE')
 assert sha(HERE/'PREREG_v1.json')==PREREG and sha(HERE/'PARENT_SOURCES_v1.json')==MANIFEST
 review=read(HERE/'ROOT_QUALIFY_REVIEW_v1.json');recipe=read(HERE/'PARENT_QUALIFY_RECIPE_v1.json')
 assert review['successful'] and review['one_qualify_authorized'] and review['recipe_sha256']==sha(HERE/'PARENT_QUALIFY_RECIPE_v1.json')
 for name,pin in recipe['files'].items():assert sha(ROOT/name)==pin,name
 assert sha(__file__)==recipe['qualifier_sha256'] and recipe['source_manifest_sha256']==MANIFEST
 build=read(HERE/'PARENT_BUILD_FINAL_v1.json');assert sha(HERE/'PARENT_BUILD_FINAL_v1.json')=='3155740b01ff5cc21090dc910d309b63e2bb5b3ce7848f44b33de2c8626d5896'
 assert build['successful'] and build['source_manifest_sha256']==MANIFEST
 guard=Path(build['guard_path']);assert sha(guard)==build['guard_sha256'];g=read(guard)
 assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors'] and g['guard_sha256']==GUARD
 assert g['max_owned_swap_bytes']==0 and all(v==0 for v in g['cgroup_events_final'].values())
 out=HERE/'parent_qualification_v1';out.mkdir(exist_ok=False)
 with (out/'NUMPY_FREE_IMPORT_v1.log').open('x') as log:
  subprocess.run(['/home/arcworker/arc-envs/client/bin/python','-B',str(HERE/'check_parent_sources_import_v1.py'),
   '--manifest-sha256',MANIFEST,'--out',str(out/'NUMPY_FREE_IMPORT_v1.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
 assert read(out/'NUMPY_FREE_IMPORT_v1.json')['successful']
 families=read(HERE/'PARENT_SOURCES_v1.json')['families'];assert len(families)==6 and families[-1]['key']=='full90'
 parent_source=ROOT/families[-1]['science']['path'];assert sha(parent_source)==PARENT
 parent_literal,_=f.literals_and_functions(parent_source.read_text());parent_params=f.decode(parent_literal['_GNN_LITERAL'])
 code,constants=gp.current_functions(parent_source.read_text());ORACLE=oracle();rng=np.random.default_rng(202610211013)
 rows=[];observer_report=None;all_case_count=0
 for n in (16,1024):
  weights=rng.normal(0,np.sqrt(2/n),(n,n)).astype('f4');state=rng.normal(0,.3,(n,64)).astype('f4');sigma=rng.uniform(.2,1.2,n).astype('f4')
  inputs={l:(rng.normal(0,.5,(n,30))*parent_params['scales'][l]+parent_params['centers'][l]).astype('f4') for l in range(1,16)}
  raw={}
  for l in range(1,16):
   mask=(np.arange(n)%5!=l%5).astype('f8')
   hc={key:(rng.normal(size=n),None if j%2 else mask,n if j%2 else int(mask.sum())) for j,key in enumerate(constants['_HC_COLUMNS']) if not(l%3==0 and j==4)}
   hcf={constants['_HCF_COLUMNS'][0]:(rng.normal(size=n),mask,int(mask.sum()))} if l==15 else {}
   raw[l]=hc,hcf
  fixture=weights,state,sigma,inputs,raw
  for terminal in (False,True):
   reference=None
   for family in [families[-1]]+families[:-1]:
    source=ROOT/family['science']['path'];assert sha(source)==family['science']['sha256']
    lit,_=f.literals_and_functions(source.read_text());params=f.decode(lit['_GNN_LITERAL']);own_code,own_constants=gp.current_functions(source.read_text());assert own_constants==constants
    report=geometry(family,params,own_code,constants,fixture,n,terminal,reference)
    if reference is None:reference=report
    else:
     assert report['records']==reference['records']
     assert report['parameter_decode_flops']==reference['parameter_decode_flops'] and report['parameter_decode_calls']==reference['parameter_decode_calls']
    rows.append({k:v for k,v in report.items() if k not in ('states','deltas','capture_proofs')});all_case_count+=1
   if n==1024 and terminal:
    observed=geometry(families[-1],parent_params,code,constants,fixture,n,terminal,reference,True)
    assert [r['flops'] for r in observed['records']]==[r['flops'] for r in reference['records']]
    assert observed['parameter_decode_flops']==reference['parameter_decode_flops']
    observer_report={k:v for k,v in observed.items() if k not in ('states','deltas')}
    observer_report['diagnostic_call_delta']=sum(r['calls'] for r in observed['records'])-sum(r['calls'] for r in reference['records'])
    observer_report['charged_flops_exact']=True
   print(json.dumps(dict(utc=utc(),n=n,terminal_hc_present=terminal,all_six_official_geometry_passed=True)),flush=True)
 baseline,bpin=gp.native_bills(RUNS/'edge8_local90_per_layer960_markline32_blas4_v1','434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4')
 paid,ppin=gp.native_bills(RUNS/'first128_paid_nocensus_native32_v1',PARENT);assert set(baseline)==set(paid)
 bills=[dict(rank=r,verified466_flops=baseline[r],measured_paid49bd_flops=paid[r],paid_bill_pct=100*(paid[r]/baseline[r]-1),new_same_geometry_head_update_flops=0) for r in sorted(paid)]
 assert all_case_count==24 and observer_report and len(observer_report['capture_proofs'])==30
 result=dict(utc=utc(),successful=True,source_manifest_sha256=MANIFEST,prereg_sha256=PREREG,script_sha256=sha(__file__),recipe_sha256=sha(HERE/'PARENT_QUALIFY_RECIPE_v1.json'),
  source_import_sha256=sha(out/'NUMPY_FREE_IMPORT_v1.json'),build_guard_sha256=sha(guard),official_version=fs.__version__,rows=rows,observer=observer_report,
  all_six_official_cold_warm_arithmetic_identical=True,all15_layers_two_geometries_both_HC_cases=True,independent_oracle_n16_all_six=True,
  own_fixed_input_state_and_message_input_bytes_unchanged=True,added_flops_vs_paid49bd=0,added_charged_calls_vs_paid49bd=0,
  measured_paid_bill_rows=bills,mean_existing_paid_bill_pct=float(np.mean([r['paid_bill_pct'] for r in bills])),verified466_bill_custody=bpin,paid49bd_bill_custody=ppin,
  no_truth_or_capture_arrays_read=True,no_full_network_prediction=True,no_native_accuracy_or_runtime_claim=True,no_adjusted_gain_claim=True,
  limits=['Synthetic observer requests/materialization are diagnostic only; actual same-source anchors remain mandatory.',
          'Same geometry has zero new operations; final changed recurrent source still needs its own full-stack bills and runtime.',
          'All existing paid49bd bill debit is retained; no arithmetic result is a banked accuracy gain.'])
 write(out/'QUALIFIED_v1.json',result)
 print(json.dumps(dict(utc=result['utc'],successful=True,all_six_arithmetic_identical=True,observer30_layers_pass=True,existing_paid_bill_pct=result['mean_existing_paid_bill_pct'])),flush=True)
if __name__=='__main__':main()
