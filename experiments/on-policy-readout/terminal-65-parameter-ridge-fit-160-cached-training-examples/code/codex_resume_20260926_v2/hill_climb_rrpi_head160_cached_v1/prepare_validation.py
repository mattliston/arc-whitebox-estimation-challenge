"""Bind the single fixed cached-head fit to the predeclared native screen."""
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
assert plan['schema']=='head160-native-validation-plan-v1'
for r in plan['pins']:assert pin(r['path'])==r,r['path']
assert plan['prepare']==pin(__file__)
p=checked(plan['protocol']); idx=checked(plan['baseline_index'])
fitpin=pin(D/'fit_result.json'); result=checked(fitpin)
assert result['completed'] is True and result['passed'] is True and result['validation_read'] is False
assert result['protocol']==plan['fit_protocol'] and result['data_index']==plan['fit_data_index']
assert result['ridge']==1e-3 and result['ranks']==p['training_ranks']
fitguardpin=pin(D/'fit_guard.json'); fitguard=checked(fitguardpin)
assert fitguard['command']==plan['fit_command']
assert fitguard['completed'] is True and fitguard['whole_group_retired'] is True
assert fitguard['exit_code']==0 and fitguard['error'] is None and fitguard['remaining_rows']==[]
assert fitguard['rss_trigger_bytes']==plan['fit_limits']['rss_bytes']
assert 0 < fitguard['max_rss_bytes'] < plan['fit_limits']['rss_bytes']
assert 0 < fitguard['elapsed_s'] <= plan['fit_limits']['seconds']
assert fitguard['max_swap_growth_bytes']==plan['fit_limits']['added_swap_bytes']==0
exportpin=pin(D/'export_manifest.json'); ex=checked(exportpin)
assert ex['passed'] is True and ex['fit_result']==fitpin
assert ex['protocol']==plan['fit_protocol'] and ex['baseline']==p['baseline']
assert ex['exporter']==plan['fit_driver']
assert result['candidate']==ex['candidate'] and result['baseline']==p['baseline']
assert result['allowed_parameter_count']==ex['allowed_parameter_count']==65
for k in ('source_byte_inverse_exact','all_other_decoded_parameter_bytes_exact','executable_ast_unchanged'):
 assert result[k] is True and ex[k] is True

candidate=ex['candidate'];assert pin(candidate['path'])==candidate
data=checked(plan['data_review'])['validation_data']
data=[dict(rank=r['rank'],weights=r['weights'],truth=r['truth']) for r in data]
assert [r['rank'] for r in data]==plan['ranks']
for r in data:
 for k in ['weights','truth']:assert pin(r[k]['path'])==r[k],r[k]['path']
for r in idx['data']:
 for k in ['prediction','truth','driver_truth']:assert pin(r[k]['path'])==r[k]
cfg=dict(schema='head160-native-validation-v1',
 candidate=candidate,baseline=p['official'],research_parent=p['baseline'],
 protocol=plan['protocol'],validation_plan=planpin,baseline_index=plan['baseline_index'],
 data_review=plan['data_review'],fit_result=fitpin,export_manifest=exportpin,
 fit_protocol=plan['fit_protocol'],fit_data_index=plan['fit_data_index'],fit_driver=plan['fit_driver'],
 fit_guard=fitguardpin,fit_command=plan['fit_command'],fit_limits=plan['fit_limits'],
 driver=plan['driver'],assessor=plan['assessor'],common=plan['common'],
 guard_source=plan['guard_source'],runtime=plan['runtime'],
 ranks=plan['ranks'],gates2=p['validation']['screen2_gates'],
 gates8=p['validation']['screen8_gates'],setup_counts=plan['setup_counts'],
 guard_limits=plan['guard_limits'],data=data,
 output=str(D/'native_validation'),native_guard=str(D/'native_guard.json'),
 assessment_output=str(D/'native_assessment.json'),
 pins=plan['pins']+[planpin,fitpin,exportpin,fitguardpin])
cfgpath=D/'native_config.json'
assert not cfgpath.exists() and not Path(cfg['output']).exists()
with cfgpath.open('x') as f:json.dump(cfg,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(config=pin(cfgpath),candidate=candidate)),flush=True)
