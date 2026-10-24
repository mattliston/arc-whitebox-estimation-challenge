"""Metadata-only provenance and stage-one source seal; no numerical imports."""
import ast
from pathlib import Path
from common import HERE,D,P2,SOURCE,PARENT_SHA,TRAIN,CHECK,sha,read,write

assert sha(SOURCE)==PARENT_SHA
assert not (HERE/'inputs.json').exists() and not (HERE/'capture_sources_sealed.json').exists()
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=store/'manifest.json';rows=read(manifest)['roles']['observer_fit']['row_manifests']
pins={}
def pin(path,expected=None):
    path=Path(path).resolve();actual=sha(path)
    if expected is not None:assert actual==expected,str(path)
    pins[str(path)]=actual
    return actual
def watch(path):
    g=read(path)
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    pin(path)
def weights(rank,expected):
    row=rows[rank];assert row['role_rank']==rank
    rp=store/row['path'];pin(rp,row['sha256']);meta=read(rp)
    assert meta['files']['weights.npy']['sha256']==expected
    wp=rp.parent/'weights.npy';assert wp.stat().st_size==meta['files']['weights.npy']['bytes']
    return str(wp)
def output_row(rank,wp_sha,prediction,pred_sha,kind,F):
    pin(prediction,pred_sha)
    return dict(rank=rank,weights=weights(rank,wp_sha),weights_sha256=wp_sha,
                prediction=str(prediction),prediction_sha256=pred_sha,prediction_kind=kind,flops=F)

qfile=D/'lazy_rotation_v1/dev/result.json';q=read(qfile);pin(qfile)
watch(D/'lazy_rotation_v1/dev_watch.json');assert not q['truth_accessed']
qualification=[output_row(r['rank'],r['weights_sha256'],r['output'],r['output_sha256'],'npy',r['flops']) for r in q['records']]
assert [r['rank'] for r in qualification]==[35,100,35]
trainfile=D/'twohop_terminal_v1/train_capture/manifest.json';train=read(trainfile);pin(trainfile)
assert train['completed'] and not train['truth_accessed'] and train['train']==TRAIN
cold={}
for r in train['rows']:
    rank=r['rank'];rr=r['record'];assert rr['candidate_sha256']==PARENT_SHA and rr['includes_initialization']
    watch(r['watch']);pin(r['completion'],r['completion_sha256'])
    cold[str(rank)]=output_row(rank,rr['weights_sha256'],rr['capture'],rr['capture_sha256'],'npz',rr['parent_flops'])
for rank in CHECK:
    path=D/f'lazy_rotation_objective_v1/confirmation/candidate_{rank}/result.json';j=read(path);pin(path)
    assert j['candidate_sha256']==PARENT_SHA and not j['truth_accessed'] and len(j['records'])==1
    rr=j['records'][0];assert rr['rank']==rank and rr['includes_initialization']
    assert rr['weight5_cache']['completed'] and rr['weight5_cache']['numeric_cache_retired']
    watch(D/f'lazy_rotation_objective_v1/confirmation/candidate_{rank}_watch.json')
    cold[str(rank)]=output_row(rank,rr['weights_sha256'],rr['output'],rr['output_sha256'],'npy',rr['flops'])
assert set(map(int,cold))==set(TRAIN+CHECK)
write(HERE/'inputs.json',dict(parent_source_sha256=PARENT_SHA,qualification=qualification,cold=cold,
                            train=TRAIN,check=CHECK,truth_accessed=False,weights_hashed_again_in_each_worker=True))
for path in HERE.iterdir():
    if path.suffix=='.py':ast.parse(path.read_text())
    if path.name in ('common.py','capture.py','replay_fnp.py','prepare_inputs.py','PROTOCOL.md','inputs.json'):pin(path)
for path in [SOURCE,D/'run_guarded.py',manifest,
    P2/'astra_phase2_reexamination_v1/THIRD_RECENTER_ROUND_V1_CLOSED.md',
    P2/'astra_phase2_reexamination_v1/current_v29_temporal_observer_v1/TRAIN_SCREEN_FAILED_20260912.md',
    P2/'astra_phase2_reexamination_v1/JOINT_PREFIX_FINAL_FIT_V1_CLOSED.md']:
    pin(path)
write(HERE/'capture_sources_sealed.json',dict(stage='capture_and_fnp_replay',truth_accessed=False,
    pins=[dict(path=k,sha256=v) for k,v in sorted(pins.items())]))
print({'qualification':len(qualification),'cold':len(cold),'pins':len(pins),'seal':sha(HERE/'capture_sources_sealed.json')})
