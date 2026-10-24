"""Read-only fixed TRAIN4 point result; true loss and distillation stay separate."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

H=Path(__file__).resolve().parent
D=H/'point_train4_v1'
source=json.loads((H/'GRADIENT_MANIFEST_v1.json').read_text())
point=json.loads((D/'FORWARDS_v1.json').read_text())
assert point['status']=='FOUR_POINT_FORWARDS_COMPLETE_AND_RETIRED'
assert [x['rank'] for x in point['rows']]==list(range(7100268,7100272))
old={x['rank']:x for x in source['rows']}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for row in point['rows']:
    seed=row['rank'];b=old[seed]
    assert sha(row['result'])==row['result_sha256']
    assert sha(b['teacher'])==b['teacher_sha256']
    r=json.loads(Path(row['result']).read_text());t=json.loads(Path(b['teacher']).read_text())
    assert sha(t['target'])==t['target_sha256']
    with np.load(t['target'],allow_pickle=False) as z:teacher=z['prediction'].copy()
    prediction=np.asarray(r['prediction'],dtype=np.float64)
    assert prediction.shape==teacher.shape==(16,1024)
    assert np.array_equal(prediction[:7],teacher[:7])
    truth_path=next(Path(p) for p in b['data_pins'] if 'truth_seed' in p)
    assert sha(truth_path)==b['data_pins'][str(truth_path)]
    with np.load(truth_path,allow_pickle=False) as z:
        noise=float(np.mean(z['mc_post_var'][-1])/int(z['N']))
        actual=float(np.mean((prediction[-1]-z['mc_post'][-1])**2))
        N=int(z['N'])
    assert np.isclose(actual,r['mse'],rtol=1e-12,atol=0)
    discrepancy=float(np.mean((prediction[-1]-teacher[-1])**2))
    pnc=actual-noise;snc=b['true_mc_mse']-noise;tnc=b['teacher_true_mc_mse']-noise
    assert min(pnc,snc,tnc)>0
    rows.append(dict(rank=seed,N=N,mc_noise=noise,
        point_teacher_discrepancy=discrepancy,unadapted_teacher_discrepancy=b['teacher_loss'],
        exact_teacher_self_discrepancy=0.,teacher_loss_change_pct=100*(discrepancy/b['teacher_loss']-1),
        point_true_mc_mse=actual,student_true_mc_mse=b['true_mc_mse'],teacher_true_mc_mse=b['teacher_true_mc_mse'],
        point_nc_mse=pnc,student_nc_mse=snc,teacher_nc_mse=tnc,
        nc_point_vs_student_pct=100*(pnc/snc-1),nc_point_vs_teacher_pct=100*(pnc/tnc-1),
        mc_point_vs_student_pct=100*(actual/b['true_mc_mse']-1),mc_point_vs_teacher_pct=100*(actual/b['teacher_true_mc_mse']-1),
        prefix_L0_6_exact=True))
def pooled(a,b):return 100*(sum(x[a] for x in rows)/sum(x[b] for x in rows)-1)
def geo(a,b):return 100*(math.exp(sum(math.log(x[a]/x[b]) for x in rows)/len(rows))-1)
fullgeo=geo('point_nc_mse','teacher_nc_mse')
result=dict(status='FIXED_TRAIN4_POINT_COMPLETE_NO_CHECK_DECISION',rows=rows,
    pooled_teacher_discrepancy_change_pct=pooled('point_teacher_discrepancy','unadapted_teacher_discrepancy'),
    geometric_teacher_discrepancy_change_pct=geo('point_teacher_discrepancy','unadapted_teacher_discrepancy'),
    pooled_nc_point_vs_student_pct=pooled('point_nc_mse','student_nc_mse'),
    geometric_nc_point_vs_student_pct=geo('point_nc_mse','student_nc_mse'),
    pooled_nc_point_vs_teacher_pct=pooled('point_nc_mse','teacher_nc_mse'),
    geometric_nc_point_vs_teacher_pct=fullgeo,
    teacher_discrepancy_wins=sum(x['teacher_loss_change_pct']<0 for x in rows),
    true_nc_wins_vs_student=sum(x['nc_point_vs_student_pct']<0 for x in rows),
    true_nc_wins_vs_teacher=sum(x['nc_point_vs_teacher_pct']<0 for x in rows),
    conditional_score_at_rank35_bill_and_provisional_margin=1.4468565569762618e-9*(1+fullgeo/100)*.9916390388660494*1.0025846,
    caveat='Training data only; no CHECK/native gain claim. Cost factor is a single-rank diagnostic plus unvalidated runtime stack.',
    point_sha256=point['point_sha256'])
with (D/'GRADE_v1.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
