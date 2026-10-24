"""Replace one diagnostic GC census, retaining all collection and model behavior."""
import ast,copy,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;D=H.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parents={'paid':('runtime_smoff_npxoff_first128_v1',{'science':'fe8cadf30daf0921a86acab95ec91348a565ce64f52e3dd4522a1936ac60620b','release':'d90e5299291fa36b53df6d508e998f2bb7819b662ad436524d56bf7b8cc0999f'}),'zero_sync':('runtime_first128_dcr_slr_v1',{'science':'58c02395cd6419ae79bbd371c912ca44bd6d45f7552d93a63c087da5a5d141f2','release':'d3991b2ffd6bf3f5ed4375257972d813cac4da45a27682bc11649f368c6ce0b7'})}
old="_GCZ_STATS.update(predicts=_GCZ_STATS['predicts'] + 1, frozen_objects=_setup_gc.get_freeze_count())"
new="_GCZ_STATS.update(predicts=_GCZ_STATS['predicts'] + 1, frozen_objects=None)"
outputs={};proof=[]
for key,(directory,roles) in parents.items():
 data=json.loads((D/directory/'BUILD_v1.json').read_text());outputs[key]={}
 for role,expected in roles.items():
  parent=Path(data['outputs']['first128'][role]['path']);assert sha(parent)==expected
  text=parent.read_text();assert text.count(old)==1;changed=text.replace(old,new)
  before=ast.parse(text);after=ast.parse(changed);fn=[n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=='_gcz_predict_v1'];assert len(fn)==1
  kw=[n for n in ast.walk(fn[0]) if isinstance(n,ast.keyword) and n.arg=='frozen_objects'];assert len(kw)==1 and ast.unparse(kw[0].value)=='_setup_gc.get_freeze_count()';kw[0].value=ast.Constant(value=None)
  assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
  assert ast.unparse(after.body[-1])=='_FAST_RESCUE_FRAC = {}'
  target=H/f'candidate_first128_{key}_nocensus_{role}_v1.py';compile(changed,str(target),'exec')
  with target.open('x')as f:f.write(changed)
  outputs[key][role]=dict(path=str(target),sha256=sha(target),parent_path=str(parent),parent_sha256=expected)
  proof.append(dict(key=key,role=role,exact_single_argument_change=True,changed_function='_gcz_predict_v1',changed_line=fn[0].lineno+5,old_argument='_setup_gc.get_freeze_count()',new_argument='None',all_other_ast_identical=True,full_compile=True,permanent_empty_rescue_map=True))
report=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),status='SOURCE_FROZEN_IMPORT_PENDING_NO_NATIVE',outputs=outputs,proof=proof,change='Omit only per-prediction read-only frozen-object census; frozen_objects=None explicitly means unmeasured. Keep predict counter/update, freeze, collect, thresholds, operation order, wrapper identities and all guards.',diagnostic_only_metadata_field=True,gc_policy_unchanged=True,numerical_source_unchanged=True,no_fnp_operations_changed=True,no_setup_or_prediction_run=True,no_native=True,no_submission=True,evidence='V/profile_first128_d399_v2/INDEPENDENT_AUDIT_v1.json18f7e42e and exact qualified profile get_freeze_count8.580769ms; instrumented duration is not runtime gain.')
with(H/'BUILD_v1.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(build_sha256=sha(H/'BUILD_v1.json'),outputs=outputs)))
