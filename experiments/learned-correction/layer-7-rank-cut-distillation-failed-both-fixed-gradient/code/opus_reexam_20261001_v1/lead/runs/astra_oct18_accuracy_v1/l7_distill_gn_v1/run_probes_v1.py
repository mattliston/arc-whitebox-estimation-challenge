"""Fixed 4 baseline + 16 small probes, one process at a time; no fitting."""
from pathlib import Path
import hashlib,json,os,subprocess
H=Path(__file__).resolve().parent
plan=json.loads((H/'PLAN_v1.json').read_text())
basis=json.loads((H/'basis_v1/BASIS_v1.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ROOT=Path(plan['repository']);out=H/'forwards_v1';out.mkdir()
env=dict(os.environ,TRAINER_THREADS='1',TRAINER_PL='1',TRAINER_CARD='M',
         TRAINER_RELIEF='1',TRAINER_STAR_ATTACH='0',TRAINER_BIRTH_ATTACH='0')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
          'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):env[k]='1'
rows=[]
try:
    for rank in plan['ranks']:
        for arm in [dict(name='base',arrays=basis['base'],sha256=basis['base_sha256']),
                    *[dict(name='d'+str(i),**v) for i,v in enumerate(basis['rows'])]]:
            for path,digest in plan['source_pins'].items():assert sha(path)==digest,path
            assert sha(arm['arrays'])==arm['sha256']
            target=out/f'{rank}_{arm["name"]}';guard=out/f'GUARD_{rank}_{arm["name"]}_v1.json'
            cmd=[plan['launcher'],'-B',plan['guard'],'--record',str(guard),'--seconds','180',
                 '--footprint-limit','9000000000','--rss-backstop','9000000000',
                 '/usr/bin/nice','-n','10',plan['torch'],'-B',plan['forward'],
                 '--rank',str(rank),'--mode','eval','--output',str(target),'--w3module',plan['w3module'],
                 '--late-arrays',arm['arrays'],'--late-arrays-sha256',arm['sha256']]
            print(json.dumps(dict(event='START',rank=rank,arm=arm['name'])),flush=True)
            with (out/f'LOG_{rank}_{arm["name"]}.txt').open('x') as log:
                child=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            assert child.returncode==0,(rank,arm['name'],child.returncode)
            g=json.loads(guard.read_text());c=json.loads((target/'COMPLETE_v1.json').read_text())
            assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0
            assert not g['remaining_rows'] and not g['error'] and g['min_pressure_free_pct']>=50
            assert max(g['max_rss_bytes'],g['max_footprint_bytes'],c['peak_rss'])<9000000000
            assert c['rank']==rank and c['mode']=='eval' and c['late_arrays_sha256']==arm['sha256']
            assert c['forward_harness_sha256']==plan['source_pins'][plan['forward']]
            assert sha(c['result'])==c['result_sha256']
            row=dict(rank=rank,arm=arm['name'],arrays_sha256=arm['sha256'],result=c['result'],
                     result_sha256=c['result_sha256'],guard=str(guard),guard_sha256=sha(guard),
                     peak_rss=c['peak_rss'],seconds=c['seconds'])
            rows.append(row)
            print(json.dumps(dict(event='RETIRED',rank=rank,arm=arm['name'],seconds=c['seconds'])),flush=True)
    with (out/'MANIFEST_v1.json').open('x') as f:
        json.dump(dict(status='FIXED_TWENTY_COMPLETE_AND_RETIRED',rows=rows,
                       basis_sha256=sha(H/'basis_v1/BASIS_v1.json'),plan_sha256=sha(H/'PLAN_v1.json')),f,indent=2)
    print('ALL_TWENTY_COMPLETE',flush=True)
except BaseException as e:
    with (out/'STOP_v1.json').open('x') as f:json.dump(dict(error=repr(e),completed=len(rows)),f,indent=2)
    raise
