"""DP lower workload for a genuinely different symmetric-output compiler."""
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/'symmetric_triangle_query_cost_v1.json'
assert not OUT.exists()
coef = json.loads((HERE/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
bank = json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']
old = json.loads((HERE/'cumulative_full_query_cost_v1.json').read_text())


@lru_cache(None)
def mm(m,k,n):
    assert all(x%32==0 and x>0 for x in (m,k,n))
    candidates = [(2*m*k*n-m*n,0)]
    candidates.extend((sum(x*y for x,y in zip(coef[str(depth)],(m*k*n,m*k,k*n,m*n))),depth)
                      for depth in (3,4,5))
    flops,depth = min(candidates)
    return dict(flops=float(flops),kernel_depth=depth)


@lru_cache(None)
def sym(m,k):
    leaf = mm(m,k,m)
    best = dict(flops=leaf['flops'],matmul_calls=1,tree_depth=0,split=None,kernel_depth=leaf['kernel_depth'])
    for s in range(32,m//2+1,32):
        a,b,off = sym(s,k),sym(m-s,k),mm(s,k,m-s)
        candidate = dict(flops=a['flops']+b['flops']+off['flops'],
                         matmul_calls=a['matmul_calls']+b['matmul_calls']+1,
                         tree_depth=1+max(a['tree_depth'],b['tree_depth']),split=s)
        if (candidate['flops'],candidate['matmul_calls'],candidate['tree_depth']) < (best['flops'],best['matmul_calls'],best['tree_depth']):
            best = candidate
    return best


def mixed(a,o):
    return 2*mm(o,a,a)['flops']+2*mm(a,a,o)['flops']+4*mm(o,a,o)['flops']


rows = []
for rank in (35,100):
    active = np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'/'alive.npy').sum(1).astype(int)
    post,next_width = int(active[12]),int(active[13])
    layers = []
    for l in range(13):
        a = int(active[l])
        general = 0. if l==0 else mixed(a,post)+mixed(a,next_width)
        source_gram,post_gram = sym(a,a),sym(post,a)
        triangle = 4*(source_gram['flops']+mm(a,a,post)['flops']+post_gram['flops'])
        layers.append(dict(source_layer=l,source_active=a,mixed_dual_flops=general,
                           triangle_flops=triangle,source_symmetric_product=source_gram,
                           post_symmetric_product=post_gram,
                           triangle_matmul_calls=4*(source_gram['matmul_calls']+1+post_gram['matmul_calls']),
                           subtotal=general+triangle))
    subtotal = sum(r['subtotal'] for r in layers)
    baseline = next(r['flops'] for r in bank if r['rank']==rank)
    oldrow = next(r for r in old['rows'] if r['rank']==rank)
    frac = subtotal/baseline
    rows.append(dict(rank=rank,old_added_fraction=oldrow['optimistic_added_fraction'],
                     improved_optimistic_added_flops=subtotal,whole_bank_measured_flops=baseline,
                     improved_optimistic_added_fraction=frac,
                     triangle_matmul_calls=sum(r['triangle_matmul_calls'] for r in layers),
                     passed=bool(frac<=.50),layers=layers))
result = dict(passed=all(r['passed'] for r in rows),threshold=.50,rows=rows,target_access=False,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              utc_finished=datetime.now(timezone.utc).isoformat(),
              qualification='Optimistic changed-compiler subtotal: symmetry DP and algebraically zero L0 mixed-source removal. Same source/query schedule and q4 triangle rank. Input symmetrization fidelity, complete other sources, Jacobians, source construction, copies and remote/runtime costs NOT certified or included. Not a candidate score or a universal complexity bound.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**{k:v for k,v in result.items() if k!='rows'},'rows':[{k:v for k,v in r.items() if k!='layers'} for r in rows]},indent=2),flush=True)
