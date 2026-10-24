from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((HERE/'protocol.json').read_text())
seal=json.loads((HERE/'seal.json').read_text())
for name,digest in seal['sources'].items():assert sha(HERE/name)==digest
rows=[];guards=[]
for group in (0,1):
    cp=HERE/f'capture_{group:02d}/complete.json';gp=HERE/f'capture_{group:02d}_guard.json'
    c,g=json.loads(cp.read_text()),json.loads(gp.read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    assert not c['targets_accessed'] and c['seal_sha256']==sha(HERE/'seal.json')
    for row in c['rows']:
        path=HERE/row['path']
        assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
    rows+=c['rows'];guards.append(g)
assert [r['rank'] for r in rows]==protocol['ranks']
assert sum(r['independent_all16_and_bill_exact'] is True for r in rows)==8
assert sum(r['bytes'] for r in rows)<protocol['max_artifact_bytes']
result=dict(rows=rows,all_groups_retired=True,added_swap_bytes=0,targets_accessed=False,
            independent_controls_exact=8,peak_rss_bytes=max(g['max_rss_bytes'] for g in guards),
            capture_seconds=sum(g['elapsed_s'] for g in guards),seal_sha256=sha(HERE/'seal.json'),
            bytes=sum(r['bytes'] for r in rows),free_disk_bytes=shutil.disk_usage(HERE).free)
with (HERE/'capture_complete.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
