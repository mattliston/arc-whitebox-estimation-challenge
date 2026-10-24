import ast,datetime,json
from pathlib import Path
from common import HERE,D,P2,SOURCE,PARENT_SHA,STORE,sha,read,write
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
assert sha(SOURCE)==PARENT_SHA
assert not (HERE/'sources_sealed.json').exists()
base=read(D/'full_lower_moment_support_v1/inputs.json')
inputs=dict(truths=base['truths'],weights=base['weights'],
    expected_parent_raw={k:v['raw'] for k,v in base['expected_parent'].items()})
write(HERE/'inputs.json',inputs)
external=[SOURCE,D/'run_guarded.py',D/'terminal_covariance_tangent_readout_v1/tangent.py',
          D/'terminal_covariance_information_v1/capture.py',D/'lazy_rotation_v1/dev/result.json',
          D/'lazy_rotation_v1/dev_watch.json',STORE/'manifest.json',
          P2/'astra_phase2_reexamination_v1/current_comp_common_cubic_v1/CHECK4_RESULT_20260910.md',
          P2/'astra_phase2_reexamination_v1/current_comp_common_cubic_v1/score_v1/result.json']
for r in inputs['truths'].values():assert sha(r['path'])==r['sha256'];external.append(Path(r['path']))
for r in inputs['weights'].values():assert sha(r['path'])==r['sha256'];external.append(Path(r['path']))
for r in read(D/'lazy_rotation_v1/dev/result.json')['records']:
    assert sha(r['output'])==r['output_sha256'];external.append(Path(r['output']))
own=[p for p in sorted(HERE.iterdir()) if p.is_file()]
for p in own:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    pins=[pin(p) for p in own+list(dict.fromkeys(external))],own_sources=len(own),before_numerical_execution=True,
    before_target_decode=True,old_common_cubic_negative_preserved=True))
print(json.dumps(dict(source_seal_sha256=sha(HERE/'sources_sealed.json'),own_sources=len(own),external_pins=len(set(external)))))

