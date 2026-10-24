"""Complete fixed deployment metadata from measured prices; no kernel execution."""
from pathlib import Path
import collections,hashlib,json
HERE=Path(__file__).resolve().parent;R=HERE.parent
chosen=json.loads((HERE/'DISPATCH_matched8_each_750_v1.json').read_text())
old=json.loads((R/'astra_crunch_compute_v1/peel_sm40_v1/TARIFF_v1.json').read_text())
tails=json.loads((HERE/'TARIFF_v1.json').read_text())
by={(x['kind'],tuple(x['shape'])):x for x in old['rows']}
tb={tuple(x['shape']):x for x in tails['tails']}
rows=[];families=collections.defaultdict(list)
for o in chosen['rows']:
    original=by[o['kind'],tuple(o['shape'])]
    body=next(t for t in original['trials'] if t['cuts']==o['cuts'])
    r=dict(o)
    m,k,n=o['shape'][:3];rank=o['shape'][3]if o['kind']=='read'else 0
    r.update(product_shape=[m,k+rank,n],body_record=body['record'],body_depth=body['L'],
             body_batch=body['P'],body_child=body['child'],arena_elements=body['arena_elements'],
             body_family=[body['L'],body['P']]+[x//(1<<body['L'])for x in body['child']],
             base_flops=original['base']['flops'],base_requests=original['base']['requests'])
    completed=[]
    for p in o['tail_plans']:
        v=tb[tuple(p['shape'])];method=p['method'];z=dict(p)
        z['forward_depths']=(sorted({d[0]for d in v['parent_forward_depths']})if method=='parent'else[v['lower_depth']]if method=='shallower'else[0])
        z['primitive']=('M._mm'if method=='parent'else'M._P6_OLD_IPAB after _TR_LAST[0]=None'if method=='shallower'else'fnp.matmul')
        z['tail_flops']=v[method]['flops'];z['tail_requests']=v[method]['requests']
        if method=='shallower':z['lower_depth']=v['lower_depth']
        completed.append(z)
    r['tail_plans']=completed
    assert r['base_flops']-r['flops']==r['saving']
    assert r['requests']-r['base_requests']==r['request_delta']
    rows.append(r)
    families[tuple(r['body_family'])].append(dict(kind=o['kind'],shape=o['shape'],record=body['record'],cuts=o['cuts']))
fam=[dict(family=list(k),L=k[0],P=k[1],r=k[2],k=k[3],c=k[4],arena_elements=7**k[0]*k[1]*(k[2]*k[3]+k[3]*k[4]+k[2]*k[4]),
          child=[(1<<k[0])*x for x in k[2:]],users=v)for k,v in sorted(families.items())]
out=dict(chosen,rows=rows,body_families=fam,body_family_count=len(fam),
         numeric_qualification=False,
         implementation_notes=['Body is exact current Sm40 L6 with frozen existing cutsets; P is the measured maximal legal batch.',
                               'Read key is original (response,young_columns,live,old_rank); product_shape folds old_rank into contraction width.',
                               'One shape uses lower-depth tree policy; reuse existing _P6_OLD_IPAB with _TR_LAST invalidation. No copied or duplicate kernel.',
                               'Copy each fringe output before subsequent arena use; original inner add and assemble copies remain required.',
                               'Do not prebind discarded original families. Family list covers selected table only; actual numeric pool sizing remains runtime owner task.',
                               'Tail forward_depths=[0] means dense. Parent depths list describes current existing dispatcher; no replacement there.'])
(HERE/'DEPLOY_STRICT750_v1.json').write_text(json.dumps(out,indent=2)+'\n')
(HERE/'FAMILIES_STRICT750_v1.json').write_text(json.dumps(dict(source_sha=chosen['source_sha'],families=fam),indent=2)+'\n')
manifest={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in ('DEPLOY_STRICT750_v1.json','FAMILIES_STRICT750_v1.json','DISPATCH_matched8_each_750_v1.json')}
print(json.dumps(dict(shapes=len(rows),families=len(fam),hashes=manifest,summary=chosen['summary']),indent=2))
