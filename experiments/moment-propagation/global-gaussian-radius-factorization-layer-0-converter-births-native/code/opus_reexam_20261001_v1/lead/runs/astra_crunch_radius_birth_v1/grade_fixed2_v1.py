"""Fixed global-radius35/100 grade against pinned same-SDK controls.

Reads independent cross halves only after complete candidate guard retirement.
No model, fitting, variant selection, or bank/leaderboard projection.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

D=Path(__file__).resolve().parent
R=D.parent
P=R/'astra_crunch_physics_v1/global_radius_v1'
CP=R/'astra_crunch_compute_v1/global_radius_checks_v1/PILOT_CONTROLS_v1.json'
controls=json.loads(CP.read_text())
pre=json.loads((P/'PILOT_PREREG_v1.json').read_text())
out=P/pre['candidate_output']
guard_path=P/'PILOT_GUARD_v1.json'
guard=json.loads(guard_path.read_text())
assert guard['completed'] and guard['whole_group_retired'], 'Wait for completed retired guard'
manifest=json.loads((out/'manifest.json').read_text())
assert manifest['sha256']==pre['candidate_sha256']
assert controls['control_source_sha256']==pre['parent_sha256']
assert manifest['ranks']==controls['fixed_ranks']==pre['ranks_order']==[35,100]
actual=[json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines() if s.strip()]
assert [z['rank'] for z in actual]==[35,100]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pred(path,shape):
    a=np.fromfile(path,dtype='<f4')
    assert a.size==int(np.prod(shape)) and np.isfinite(a).all()
    return a.reshape(shape).astype(np.float64)

rows=[]
for ctl,row in zip(controls['rows'],actual):
    rank=ctl['rank'];assert row['rank']==rank
    assert sha(ctl['prediction'])==ctl['prediction_sha256']
    p0=pred(ctl['prediction'],ctl['shape'])
    prediction_path=out/f'pred_{rank}.bin'
    p1=pred(prediction_path,ctl['shape'])
    truth=[]
    for t in ctl['truths']:
        assert sha(t['path'])==t['sha256']
        with np.load(t['path']) as z:truth.append(z[t['key']].astype(np.float64))
    A,B=truth
    assert A.shape==B.shape==p0.shape
    u0=np.mean((p0-A)*(p0-B),axis=1)
    u1=np.mean((p1-A)*(p1-B),axis=1)
    assert math.isclose(float(u0[-1]),ctl['control_U'],rel_tol=1e-12,abs_tol=1e-20)
    assert u0[-1]>0 and u1[-1]>0
    ur=float(u1[-1]/u0[-1]);fr=row['flops']/ctl['control_flops']
    rows.append(dict(rank=rank,phase=ctl['phase'],control_U=float(u0[-1]),candidate_U=float(u1[-1]),
        raw_ratio=ur,raw_pct=100*(ur-1),control_flops=ctl['control_flops'],candidate_flops=row['flops'],
        flop_ratio=fr,flop_pct=100*(fr-1),adjusted_ratio=ur*fr,adjusted_pct=100*(ur*fr-1),
        control_requests=ctl['control_requests'],candidate_requests=row['requests'],
        request_delta=row['requests']-ctl['control_requests'],
        control_U_by_layer=u0.tolist(),candidate_U_by_layer=u1.tolist(),
        mean_prediction_shift=float(np.mean(p1[-1]-p0[-1])),
        rms_prediction_shift=float(np.sqrt(np.mean((p1[-1]-p0[-1])**2))),
        candidate_prediction=str(prediction_path),candidate_prediction_sha256=sha(prediction_path),
        wall_s=row.get('wall_s'),residual_instructions=row.get('residual_instructions'),
        server_footprint_max=row.get('server_footprint_max'),finite=True))

ur=np.array([z['raw_ratio'] for z in rows]);ar=np.array([z['adjusted_ratio'] for z in rows])
result=dict(status='FIXED2_PAIRED_SCIENCE_RESULT_NOT_BANK',n=2,candidate_sha256=manifest['sha256'],
    control_sha256=controls['control_source_sha256'],control_manifest=str(CP),control_manifest_sha256=sha(CP),
    candidate_manifest_sha256=sha(out/'manifest.json'),guard_sha256=sha(guard_path),
    guard_completed=True,whole_group_retired=True,rows=rows,
    raw_arithmetic_mean_pct=float(100*(np.mean(ur)-1)),raw_geomean_pct=float(100*(np.exp(np.mean(np.log(ur)))-1)),
    raw_log_se_pct=float(100*np.std(np.log(ur),ddof=1)/math.sqrt(2)),
    raw_pooled_pct=100*(sum(z['candidate_U'] for z in rows)/sum(z['control_U'] for z in rows)-1),
    adjusted_arithmetic_mean_pct=float(100*(np.mean(ar)-1)),
    adjusted_geomean_pct=float(100*(np.exp(np.mean(np.log(ar)))-1)),
    adjusted_log_se_pct=float(100*np.std(np.log(ar),ddof=1)/math.sqrt(2)),
    adjusted_pooled_pct=100*(sum(z['candidate_U']*z['candidate_flops'] for z in rows)/sum(z['control_U']*z['control_flops'] for z in rows)-1),
    pooled_flop_pct=100*(sum(z['candidate_flops'] for z in rows)/sum(z['control_flops'] for z in rows)-1),
    raw_wins=int(np.sum(ur<1)),adjusted_wins=int(np.sum(ar<1)),
    uncertainty='Two fixed networks only. Independent A/B cross-product removes shared Monte Carlo noise bias; it does not make two-network generalization certain. No score/bank extrapolation.',
    guard=dict((k,guard[k]) for k in ('max_rss_bytes','max_footprint_bytes','min_pressure_free_pct') if k in guard),
    grader_sha256=sha(__file__))
(D/'FIXED2_RESULT_v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','guard')},indent=2))
for z in rows:print(json.dumps({k:v for k,v in z.items() if k not in ('candidate_U_by_layer','control_U_by_layer')},indent=2))
