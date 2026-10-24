"""Shape-only upper-value screen for rectangular schedule specialization.

Uses independently measured kernel cost polynomials and the authoritative
existing r35 matrix census. It does not execute an estimator, inspect
labels, or claim a measured new utilization. All layouts must fit the
existing 447469568-element arena; no memory increase is proposed.
"""
import json
from functools import lru_cache
from pathlib import Path
import re
from packed_arena_generic import layout

HERE = Path(__file__).resolve().parent
out = HERE / "rectangular_schedule_value_v1.json"
if out.exists():
    raise FileExistsError(out)
price = json.loads((HERE/"direct_kernel_schedule_prices_v1.json").read_text())["coefficients"]
census = json.loads((HERE/"compact_gaussfirst_cost_census_r35_v2.json").read_text())
arena = 447469568
options = []
for depth in (3, 4, 5, 6, 7):
    for tile in (2048, 4096):
        size = max(layout(1024,1024,tile,depth)[0], layout(1024,tile,1024,depth)[0])
        if size <= arena:
            options.append((depth,tile,size))


def kernel(m,k,n,depth):
    unit = 2**depth
    mp,kp,npad = ((x+unit-1)//unit*unit for x in (m,k,n))
    a,b,c,d = price[str(depth)]
    value = a*mp*kp*npad+b*mp*kp+c*kp*npad+d*mp*npad
    # Conservative zero-initialization + copy charge on padded inputs.
    if mp != m or kp != k:
        value += mp*kp+m*k
    if kp != k or npad != n:
        value += kp*npad+k*n
    return value


@lru_cache(None)
def price_shape(m,k,n,depth,tile):
    if n>tile:
        return sum(price_shape(m,k,min(tile,n-j),depth,tile) for j in range(0,n,tile))+m*n
    if k>tile:
        return sum(price_shape(m,min(tile,k-j),n,depth,tile) for j in range(0,k,tile))+(len(range(0,k,tile))-1)*m*n
    # Do not alter already optimized square schedules in this proposal.
    if max(m,k,n)<=1024:
        return min(2*m*k*n-m*n,*(kernel(m,k,n,d) for d in (3,4,5,6)))
    return min(2*m*k*n-m*n,kernel(m,k,n,depth))


rows=[]
for label, record in census["matrix_shapes"].items():
    match = re.search(r"\((\d+), (\d+)\) @ \((\d+), (\d+)\)",label)
    assert match,label
    m,k,k2,n=map(int,match.groups())
    assert k==k2
    if max(m,k,n)<=1024:
        continue
    baseline=price_shape(m,k,n,6,4096)
    candidates=[(price_shape(m,k,n,d,t),d,t,size) for d,t,size in options]
    best,d,t,size=min(candidates)
    rows.append(dict(label=label,calls=record["calls"],measured_parent_flops=record["flops"],
                     baseline_model=baseline,best_model=best,depth=d,tile=t,
                     arena_bytes=size*4,projected_saving=(baseline-best)*record["calls"]))
saving=sum(r["projected_saving"] for r in rows)
receipt=dict(options=options,rows=rows,projected_flops_saved=saving,
             fraction_of_old_census_total=saving/census["flops"],
             interpretation="Shape-only projection on Gaussian-first census; not measured candidate utilization. Exact-source gates required before use.")
out.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="rows"}),flush=True)
print(json.dumps(sorted(rows,key=lambda r:r["projected_saving"],reverse=True)[:6]),flush=True)
