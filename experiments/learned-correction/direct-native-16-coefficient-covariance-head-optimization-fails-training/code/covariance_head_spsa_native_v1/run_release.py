from pathlib import Path
from common import HERE,D,PARENT,RUNTIME,read,write,pin,verify,guarded
for p in read(HERE/'release_sources_sealed.json')['pins']:verify(p)
assert read(HERE/'outcome.json')['passed']
full=sum(len(read(p)['records']) for p in HERE.glob('*/result.json') if 'records' in read(p))
assert full+11<=120
R=D/'readout_residual_runtime_v1';rows=[]
for scope,driver in [('conservative',R/'plain_driver.py'),('prediction_only',R/'predict_only_probe_v1/driver.py')]:
    for i,rank in enumerate([35,100]):
        for role in (['parent','candidate'] if i==0 else ['candidate','parent']):
            label=f'release_{scope}_{rank}_{role}';source=PARENT if role=='parent' else HERE/'candidate.py'
            guarded(label,120,[RUNTIME,'-B',driver,source,HERE/label,str(rank)])
            rows.append(dict(scope=scope,rank=rank,role=role,receipt=pin(HERE/label/'completed.json'),watch=pin(HERE/f'{label}_watch.json')))
            c=read(HERE/label/'completed.json')['records'][0]
            print(dict(scope=scope,rank=rank,role=role,residual=c['summary']['residual_wall_time_s'],wall_s=c['wall_s']),flush=True)
label='release_warm_candidate';guarded(label,240,[RUNTIME,'-B',R/'plain_driver.py',HERE/'candidate.py',HERE/label,'35,100,35'])
write(HERE/'release_client_completed.json',dict(rows=rows,warm=pin(HERE/label/'completed.json'),full_native_calls=full,client_full_calls=11))
guarded('release_qualification',60,[RUNTIME,'-B',HERE/'qualify_release.py'])
if not read(HERE/'release_qualified.json')['passed']:
    write(HERE/'release_outcome.json',dict(passed=False,stage='runtime',uploaded=False));raise SystemExit(0)
ROOT=Path('/Users/matt/arc-white-box-estimation')
guarded('contract',60,[ROOT/'.venv/bin/whest','validate','--estimator',HERE/'candidate.py','--json'])
guarded('package',60,[ROOT/'.venv/bin/python','-B',HERE/'package.py'])
write(HERE/'release_outcome.json',dict(passed=True,stage='package',package=pin(HERE/'package.json'),uploaded=False))
