"""Record completed native8 exact component only; stdlib, no model or release action."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent;H=D.parent;P=H/'hill_climb_exact_plane_reuse_v1'
NATIVE=str(H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def check_tree(x):
 if isinstance(x,dict):
  if {'path','sha256'}<=x.keys():assert {'path','sha256','bytes'}<=x.keys() and pin(x['path'])=={k:x[k] for k in ['path','sha256','bytes']}
  for v in x.values():check_tree(v)
 elif isinstance(x,list):
  for v in x:check_tree(v)
def guard(p,seconds,limit=7500000000,cmd=None):
 g=read(p);assert g['completed'] and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] and not g['remaining_rows']
 assert g['max_swap_growth_bytes']==0 and 0<g['max_rss_bytes']<g['rss_trigger_bytes']==limit and 0<g['elapsed_s']<=seconds
 if cmd is not None:assert g['command']==cmd
 return pin(p)
a=read(D/'native8_assessment.json');c=read(a['config']['path']);b=read(D/'boundary_assessment.json');ad=read(D/'broad_admission.json');aux=read(D/'root_threecast_increment_v1.json')
for obj in [a,c,b,ad,aux]:check_tree(obj)
assert a['candidate']==pin(D/'candidate.py') and a['candidate']['sha256']=='0f7b93c4c45c447d76f04f428c1aeec066c5eb30027671653565b7c5fe3d93c3'
assert a['comparator']==pin(P/'candidate.py') and a['config']==pin(D/'native_screen_continuation.json')
assert a['passed_integrity'] and a['admitted'] and a['n']==8 and a['raw_pct']==0 and a['adjusted_wins']==8 and all(a['gates'].values())
assert a['all16_predictions_finite'] and a['all16_parent_bytes_exact'] and a['configured_parent_bytes_exact'] and a['parity_layers']==list(range(16))
assert b['passed'] and b['receipt_checks_passed'] and all(b['gates'].values()) and b['candidate']==a['candidate']
assert pin(D/'root_threecast_increment_v1.json')['sha256']=='909281332faf88bb487eace638f6d53ffa7cb77ce689cd37f0b545e599f3d7f5'
assert aux['passed'] and aux['candidate']==a['candidate'] and aux['bill_pct']<0 and all(aux['all16_prediction_file_bytes_equal'].values())
assert pin(D/'broad_admission.json')['sha256']=='f8508e1aa2aa79a684d1eadf9b6386184ea4e356d1b87b59c74c1ede70419048'
assert ad['passed'] and ad['candidate']==a['candidate'] and ad['own_fixed40_and_serving_still_required']
guards=[];parity=[]
for size in ['2','6']:
 spec=c['arms']['candidate']['blocks'][size];folder=H/spec['tag']
 guards.append(guard(spec['guard'],spec['limits']['seconds'],cmd=[NATIVE,'-B',spec['driver']['path'],a['candidate']['path'],','.join(map(str,spec['ranks'])),spec['tag']]))
 for rank in spec['ranks']:
  cp=pin(folder/f'pred_{rank}.npy');pp=c['arms']['base']['blocks'][size]['artifacts']['predictions'][str(rank)]
  assert cp['sha256']==pp['sha256'] and cp['bytes']==pp['bytes'];parity.append(dict(rank=rank,candidate=cp,parent=pp,all16_file_bytes_exact=True))
for stage in ['2','8']:
 spec=c['stages'][stage];guards.append(guard(spec['assessment_guard'],spec['assessment_limits']['seconds'],cmd=[NATIVE,'-B',c['engine']['path'],a['config']['path'],a['config']['sha256'],stage]))
q=c['boundary_before8'];guards.append(guard(q['assessment_guard'],q['assessment_limits']['seconds'],cmd=q['assessment_command']))
guards.append(guard(D/'boundary_guard.json',240));guards.append(guard(D/'primitive_guard.json',60,1000000000))
result=dict(status='RETAINED_COMPONENT',candidate=a['candidate'],comparator=a['comparator'],native8=pin(D/'native8_assessment.json'),config=a['config'],engine=a['engine'],boundary=pin(D/'boundary_assessment.json'),parent_retained=pin(P/'retained_component.json'),root_threecast_increment=pin(D/'root_threecast_increment_v1.json'),broad_admission=pin(D/'broad_admission.json'),incremental_adjusted_pct=a['adjusted_pct'],total_vs_S34_adjusted_pct=ad['total_vs_S34_native8_pct'],total_comparison_basis='Actual saved8 S34 comparison in pinned root broad admission',raw_pct=a['raw_pct'],wins=a['adjusted_wins'],confidence_interval95_pct=a['adjusted_ci95_pct'],all16_parent_bytes_exact=True,whole8_byte_parity=parity,completed_guards=guards,added_swap_bytes=0,numerical_jobs='native2, native6, native8 assessment, primitive, own boundary all completed and retired; fixed40 is separate and still required',decision='Retain exact three-cast plus row-expansion scatter research component. Fixed40 and full serving have not been claimed passed. No release or official bank change.',release_qualified=False,submitted=False,recorder=pin(__file__))
p=D/'retained_component.json'
with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(pin(p)))
