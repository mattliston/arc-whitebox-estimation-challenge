"""Exact control and independent scalar-fsum score comparison."""
from pathlib import Path
import json,hashlib,math,sys
import numpy as np
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
read=lambda p:json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
for x in read(HERE/'build_ordinary.json')['sources']:assert pin(x['path'])==x
for x in read(HERE/'endpoint_seal_ordinary.json')['sources']:assert pin(x['path'])==x
baseline=H2/'thin_star_route_train2_v1';disabled=H2/'global_pair_exact_disabled_ordinary_train2_v1'
brows=[json.loads(x) for x in (baseline/'rows.jsonl').read_text().splitlines()]
drows=[json.loads(x) for x in (disabled/'rows.jsonl').read_text().splitlines()]
assert [x['rank'] for x in brows]==[x['rank'] for x in drows]==[35,100]
assert read(disabled/'manifest.json')['sha256']==pin(HERE/'disabled_ordinary.py')['sha256']
for b,d in zip(brows,drows):
    assert b['flops']==d['flops'] and d['cf']['calls']==0
    assert np.load(baseline/f"pred_{b['rank']}.npy").tobytes()==np.load(disabled/f"pred_{b['rank']}.npy").tobytes()
g=read(HERE/'disabled_ordinary_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
if sys.argv[1]=='disabled':
    with (HERE/'disabled_qualified_ordinary.json').open('x') as f:json.dump(dict(all16_bytes_exact=True,bills_exact=True,ranks=[35,100],candidate=pin(HERE/'disabled_ordinary.py')),f,indent=2)
    print('DISABLED_ALL16_AND_BILLS_EXACT',flush=True);sys.exit(0)
raise RuntimeError('Use team_score_screen_v1.py for enabled bank comparison')
