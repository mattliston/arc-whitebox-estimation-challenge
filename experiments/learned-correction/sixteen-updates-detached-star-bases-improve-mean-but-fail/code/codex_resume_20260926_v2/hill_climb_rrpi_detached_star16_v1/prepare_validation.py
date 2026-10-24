"""Bind the single completed update16 endpoint to the predeclared native screen."""
from pathlib import Path
import hashlib, json, sys
D=Path(__file__).resolve().parent
def pin(path):
 p=Path(path).resolve(); h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
def checked(r):
 assert pin(r['path'])==r,r['path']
 return json.loads(Path(r['path']).read_text())
assert len(sys.argv)==2
planpin=pin(D/'validation_plan.json');assert planpin['sha256']==sys.argv[1]
plan=checked(planpin)
assert plan['schema']=='detached-star16-native-validation-plan-v1'
for r in plan['pins']:assert pin(r['path'])==r,r['path']
assert plan['prepare']==pin(__file__)
p=checked(plan['protocol']); idx=checked(plan['baseline_index'])
donepin=pin(D/'training_complete.json'); done=checked(donepin)
assert done['completed'] and done['updates']==done['selected_endpoint']==16
assert done['protocol']==plan['protocol'] and len(done['segments'])==2
exportpin=pin(D/'export_manifest.json'); ex=checked(exportpin)
assert ex['passed'] and ex['training_complete16_verified'] and ex['training_guards_checked']==4
assert ex['new_training_guards_checked']==2 and ex['prior_training_complete8_verified']
assert ex['training_completion']==donepin and ex['baseline']==p['baseline']
candidate=ex['candidate'];assert pin(candidate['path'])==candidate
assert ex['exporter']==plan['exporter']
for r in ex['training_guards']+ex['segment_completions']+ex['prior_training_guards']+[ex['prior_training_completion']]:assert pin(r['path'])==r
assert ex['training_guards']==done['segments']
data=checked(plan['data_review'])['validation_data']
data=[dict(rank=r['rank'],weights=r['weights'],truth=r['truth']) for r in data]
assert [r['rank'] for r in data]==plan['ranks']
for r in data:
 for k in ['weights','truth']:assert pin(r[k]['path'])==r[k],r[k]['path']
for r in idx['data']:
 for k in ['prediction','truth','driver_truth']:assert pin(r[k]['path'])==r[k]
cfg=dict(schema='detached-star16-native-validation-v1',training_updates=16,
 candidate=candidate,baseline=p['official'],research_parent=p['baseline'],
 protocol=plan['protocol'],validation_plan=planpin,baseline_index=plan['baseline_index'],
 data_review=plan['data_review'],training_complete=donepin,export_manifest=exportpin,
 driver=plan['driver'],assessor=plan['assessor'],common=plan['common'],
 guard_source=plan['guard_source'],runtime=plan['runtime'],
 ranks=plan['ranks'],gates8=p['validation']['screen8_gates'],setup_counts=plan['setup_counts'],
 guard_limits=plan['guard_limits'],data=data,
 output=str(D/'native_validation'),native_guard=str(D/'native_guard.json'),
 assessment_output=str(D/'native_assessment.json'),
 pins=plan['pins']+[planpin,donepin,exportpin]+ex['training_guards']+ex['segment_completions'])
cfgpath=D/'native_config.json'
assert not cfgpath.exists() and not Path(cfg['output']).exists()
with cfgpath.open('x') as f:json.dump(cfg,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(config=pin(cfgpath),candidate=candidate)),flush=True)
