"""Exactly four frozen point forwards, after coherence PASS; no fitting/CHECK."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[1]
ROOT=Path('/Users/matt/arc-white-box-estimation')
OUT=H/'point_train4_v1'
POINT=R/'astra_oct18_lead_v1/l7_distill_decision_v1/point_v1/late_arrays.npz'
POINT_SHA='5898771f266be012a59ee000de5ccb51828a4050aa1f03bf6ed8d593a08c7623'
FORWARD=H/'run_forward_v1.py'
FORWARD_SHA='25a289350059f861a955e1ffdfb878b3158c10f4a8a3f02360edce82383888c6'
GUARD=H.parent/'current_torch_rtt_port_v1/offline_train8_v1/guard_offline9gb_free50_v1.py'
W3=R/'astra_oct18_lead_v1/current_torch_jsk_port_v1/torch_w3_current95c_v2.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=dict(os.environ,TRAINER_THREADS='1',TRAINER_PL='1',TRAINER_CARD='M',
         TRAINER_RELIEF='1',TRAINER_STAR_ATTACH='0',TRAINER_BIRTH_ATTACH='0')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):
    env[key]='1'
assert sha(POINT)==POINT_SHA and sha(FORWARD)==FORWARD_SHA
OUT.mkdir()
with (OUT/'PLAN_v1.json').open('x') as f:
    json.dump(dict(ranks=list(range(7100268,7100272)),point=str(POINT),point_sha256=POINT_SHA,
                   forward=str(FORWARD),forward_sha256=FORWARD_SHA,guard_bytes=9000000000,
                   source_gradients=str(H/'GRADIENT_MANIFEST_v1.json'),no_optimizer=True),f,indent=2)
rows=[]
try:
    for seed in range(7100268,7100272):
        assert sha(POINT)==POINT_SHA and sha(FORWARD)==FORWARD_SHA
        path=OUT/f'point_{seed}';guard=OUT/f'GUARD_{seed}_v1.json'
        cmd=[str(ROOT/'.venv/bin/python'),'-B',str(GUARD),'--record',str(guard),
             '--seconds','180','--footprint-limit','9000000000','--rss-backstop','9000000000',
             '/usr/bin/nice','-n','10','/Users/matt/anaconda3/bin/python','-B',str(FORWARD),
             '--rank',str(seed),'--mode','eval','--output',str(path),'--w3module',str(W3),
             '--late-arrays',str(POINT),'--late-arrays-sha256',POINT_SHA]
        print(json.dumps(dict(event='START_POINT',rank=seed)),flush=True)
        with (OUT/f'LOG_{seed}_v1.txt').open('x') as log:
            done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert done.returncode==0,(seed,done.returncode)
        g=json.loads(guard.read_text());c=json.loads((path/'COMPLETE_v1.json').read_text())
        assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0
        assert not g['error'] and not g['remaining_rows'] and g['min_pressure_free_pct']>=50
        assert max(g['max_rss_bytes'],g['max_footprint_bytes'],c['peak_rss'])<9000000000
        assert c['rank']==seed and c['mode']=='eval' and c['late_arrays_sha256']==POINT_SHA
        assert c['forward_harness_sha256']==FORWARD_SHA and sha(c['result'])==c['result_sha256']
        rows.append(dict(rank=seed,result=c['result'],result_sha256=c['result_sha256'],
                         guard=str(guard),guard_sha256=sha(guard),true_mc_mse=c['mse'],
                         seconds=c['seconds'],peak_rss=c['peak_rss']))
        print(json.dumps(dict(event='POINT_RETIRED',rank=seed,seconds=c['seconds'],peak_rss=c['peak_rss'])),flush=True)
    with (OUT/'FORWARDS_v1.json').open('x') as f:
        json.dump(dict(status='FOUR_POINT_FORWARDS_COMPLETE_AND_RETIRED',rows=rows,
                       point_sha256=POINT_SHA,parameters_updated=False),f,indent=2)
    print('FOUR_POINT_FORWARDS_COMPLETE',flush=True)
except BaseException as e:
    with (OUT/'STOP_v1.json').open('x') as f:
        json.dump(dict(error=repr(e),completed=[x['rank'] for x in rows],parameters_updated=False),f,indent=2)
    raise
