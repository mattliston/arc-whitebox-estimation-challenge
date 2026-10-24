"""Request census: per-network requests (rows.jsonl) and per-op call counts (server.json ops) vs 2b108 and b709 production boundaries."""
import json,sys
from pathlib import Path
F=Path(__file__).resolve().parent;R=F.parent
runs={'f384':F/'boundary/normal_v2','2b108':R/'astra_smirnov_rpe_delivery_v1/boundary/numerical_v1/final_normal','b709':R/'astra_table_r7_delivery_v1/boundary/numerical_v1/final_normal'}
rows={k:[json.loads(l) for l in (p/'rows.jsonl').read_text().splitlines()] for k,p in runs.items()}
for k,v in rows.items():print('%-6s'%k,' '.join('rank%d req %d flops %.4fG util %.5f residual %.3f wall %.1f'%(r['rank'],r['requests'],r['flops']/1e9,r['util'],r['residual_s'],r['wall_s']) for r in v))
def ops(p):
    for name in ('server.json','launcher.json'):
        q=p/name
        if q.exists():
            d=json.loads(q.read_text());d=d.get('server',d)
            if 'ops' in d:return {k:v[0] for k,v in d['ops'].items()}
    return None
o={k:ops(p) for k,p in runs.items()}
if all(o.values()):
    keys=sorted(set(o['f384'])|set(o['2b108']),key=lambda k:-(o['f384'].get(k,0)-o['2b108'].get(k,0)))
    print('per-op call delta over setup+2 networks (f384-2b108):',', '.join('%s %+d'%(k,o['f384'].get(k,0)-o['2b108'].get(k,0)) for k in keys if o['f384'].get(k,0)!=o['2b108'].get(k,0))[:1500])
