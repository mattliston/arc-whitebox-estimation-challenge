"""Public-accounting analytical cost gate, no target or large tensor work."""
import json
from pathlib import Path
import numpy as np
from flopscope.accounting import solve_cost

HERE=Path(__file__).resolve().parent
OUT=HERE/'quadratic_skew_covariance_cost_v1.json'
assert not OUT.exists()
bank=json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']
records=[]
for rank in (35,100):
    alive=np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'/'alive.npy')
    base=next(x['flops'] for x in bank if x['rank']==rank)
    layers=[]
    for layer in range(8,15):
        n=int(alive[layer].sum())
        solve=2*solve_cost(n,nrhs=n,symmetric=True)
        gram=2*(2*n**3-n*n)
        # Includes solve-residual check (one additional full matrix product).
        check=2*(2*n**3-n*n)
        # Three complete order2 pair responses and proxy construction, f64.
        # Full1024 matrices including normalization/diagonal/assembly.
        point=2*500*1024**2
        layers.append(dict(layer=layer,active=n,solve=solve,gram=gram,residual_check=check,pointwise_allowance=point,total=solve+gram+check+point))
    extra=sum(x['total'] for x in layers)
    records.append(dict(rank=rank,base=base,layers=layers,extra=extra,extra_fraction=extra/base,passed=extra/base<=.15))
result=dict(records=records,passed=all(x['passed'] for x in records),qualification='Analytical float64 classical costs, current fixed active tape; no native time/memory/score claim.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
