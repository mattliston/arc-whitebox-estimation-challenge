"""Metadata-only fit admission, binding completed features and public references."""
import ast
from pathlib import Path
from common import HERE,D,P2,TRAIN,CHECK,sha,read,write,seal_check
seal_check('acquisition_sources_sealed.json')
seal_check('fit_implementation_sealed.json')
assert not (HERE/'fit_sources_sealed.json').exists()
g=read(HERE/'capture_all_watch.json');m=read(HERE/'captures/manifest.json')
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert m['completed'] and not m['truth_accessed'] and [r['rank'] for r in m['rows']]==TRAIN+CHECK
source=P2/'astra_phase2_reexamination_v1/joint_prefix_final_fit_v1/manifest.json'
assert sha(source)=='d61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
targets={r['rank']:r for r in read(source)['target_sources']}
check_source=P2/'astra_phase2_reexamination_v1/birth_direction_fit_v1/terminal_score_v2/truth_bindings.json'
check_bindings=read(check_source)
weight_metadata=read(HERE/'inputs.json')['cold']
for rank in CHECK:
    assert rank not in targets
    target_path=P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
    targets[rank]=dict(rank=rank,weight_file=weight_metadata[str(rank)]['weights'],target_file=str(target_path),target_sha256=check_bindings[str(target_path)])
roster=[]
pins={r['path']:r['sha256'] for r in read(HERE/'acquisition_sources_sealed.json')['pins']}
def pin(path,expected=None):
    path=Path(path);actual=sha(path)
    if expected is not None:assert actual==expected,str(path)
    pins[str(path)]=actual
for r in m['rows']:
    rec=r['record'];rank=r['rank'];t=targets[rank]
    assert t['weight_file']==rec['weights'] and '/public_g22_operator_features_v1/truthpost/' in t['target_file']
    assert rec['includes_initialization'] and rec['all16_parent_bytes_and_bill_exact']
    gg=read(r['watch']);assert gg['completed'] and gg['whole_group_retired'] and gg['max_swap_growth_bytes']==0
    pin(r['completion'],r['completion_sha256']);pin(r['watch']);pin(rec['path'],rec['sha256']);pin(r['parent_P']['path'],r['parent_P']['sha256'])
    roster.append(dict(rank=rank,role='TRAIN' if rank in TRAIN else 'CHECK',capture=rec,
                       target_file=t['target_file'],target_sha256=t['target_sha256']))
write(HERE/'fit_inputs.json',dict(rows=roster,train=TRAIN,check=CHECK,parent_P=read(HERE/'capture_qualification/result.json')['parent_P'],
    target_manifest=dict(path=str(source),sha256=sha(source)),check_target_binding=dict(path=str(check_source),sha256=sha(check_source)),target_arrays_decoded=False))
for name in ('model.py','fit.py','prepare_fit.py','prepare_fit_v2.py'):ast.parse((HERE/name).read_text())
for name in ('model.py','fit.py','prepare_fit.py','prepare_fit_v2.py','metadata_admission_failure.json','fit_implementation_sealed.json','fit_inputs.json','capture_all.py','acquisition_sources_sealed.json','capture_all_watch.json','captures/manifest.json'):pin(HERE/name)
pin(source);pin(check_source)
for path,digest in pins.items():assert sha(path)==digest,path
write(HERE/'fit_sources_sealed.json',dict(stage='single_fixed_150_update_fit',target_arrays_decoded=False,
    pins=[dict(path=k,sha256=v) for k,v in sorted(pins.items())]))
print(dict(pins=len(pins),fit_sources_sha256=sha(HERE/'fit_sources_sealed.json')))
