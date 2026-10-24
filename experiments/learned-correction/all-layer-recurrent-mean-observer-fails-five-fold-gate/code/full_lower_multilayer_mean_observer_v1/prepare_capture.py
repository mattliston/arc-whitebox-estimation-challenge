import datetime
from pathlib import Path
from common import HERE,D,PARENT,NATIVE,TARGETS,TRAIN,CHECK,PANEL,CONFIRM,read,write,pin,verify,weights_path
target_map={r['rank']:r for r in read(TARGETS)['target_sources']}
rows=[]
for rank in TRAIN+CHECK+PANEL+CONFIRM:
    t=target_map.get(rank);wp=weights_path(rank)
    if t is None:
        # Confirmation targets are also pinned in the established edge panel manifest.
        m=read(D/'full_lower_covariance_edge_native_v1/panel_inputs.json')
        e=next(x for x in m['confirmation'] if x['rank']==rank)
        truth=e['truth'];verify(truth)
    else:
        truth=pin(t['target_file']);assert truth['sha256']==t['target_sha256']
        assert Path(t['weight_file']).resolve()==wp.resolve()
    rows.append(dict(rank=rank,weights=pin(wp),truth=truth))
parents={}
for rank in TRAIN:
    paths=[D/f'covariance_head_spsa_native_v1/baseline_{rank}/result.json',
        D/f'covariance_head_native_response_v1/baseline_replay_audit_v1/base{rank}/result.json',
        D/f'full_lower_covariance_edge_native_v1/panel_{rank}/result.json']
    path=next((p for p in paths if p.exists()),None)
    if path is not None:
        x=read(path);r=x['records'][0]
        assert r['rank']==rank and r['includes_initialization']
        parents[str(rank)]=dict(output=pin(r['output']),flops=r['flops'],weights_sha256=r['weights_sha256'],receipt=pin(path))
write(HERE/'inputs.json',dict(rows=rows,known_training_parents=parents,train=TRAIN,check=CHECK,targets_decoded=False,target_manifest=pin(TARGETS)))
files=[HERE/n for n in ['PROTOCOL.md','common.py','prepare_capture.py','capture.py','run_capture.py','qualify_capture.py','inputs.json']]+[PARENT,NATIVE,D/'run_guarded.py',D/'official_best_331684.json']
write(HERE/'capture_sources_sealed.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pins=[pin(p) for p in files],target_arrays_decoded=False))
print(dict(source_pins=len(files),known_training_parents=len(parents),missing_parent_calls=[r for r in TRAIN if str(r) not in parents]),flush=True)
