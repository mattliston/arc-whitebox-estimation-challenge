"""Conservative classical operation price for the new36-coefficient kernel."""
import json
from pathlib import Path
from flopscope.accounting import pointwise_cost,qr_cost

HERE=Path(__file__).resolve().parent;OUT=HERE/'covariance_shared_vertex_cost_v1.json';assert not OUT.exists()
n=1024
# New dense products priced CLASSICALLY in f32, without assuming Winograd.
grams=3*(2*n*n*n-n*n)
# F(3) @ Theta(3,3) @ F.T, six b,p combinations, scalar operations64.
endpoints=6*2*((2*n*3*3-n*3)+(2*n*3*n-n*n))
# Three weighted input matrices32 + full64 outer sigmas, mixed-endpoint
# scaling, sum and Hadamard powers; deliberately generous200 ops per pair.
pointwise=3*n*n+2*200*n*n
per_layer=grams+endpoints+pointwise
bank=json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']
rows=[]
for rank in (35,100):
    base=next(x['flops'] for x in bank if x['rank']==rank)
    extra=7*per_layer
    rows.append(dict(rank=rank,current_measured_flops=base,extra_flops=extra,extra_fraction=extra/base,gate_pass=extra/base<=.10))
result=dict(rows=rows,passed=all(r['gate_pass'] for r in rows),per_layer=dict(grams_f32=grams,endpoints_f64=endpoints,pointwise_allowance=pointwise),
            qualification='Analytical conservative f32 classical full GEMMs plus f64 endpoint/assembly allowance, fixed layers8..14. Not native metering, memory, latency or score. Offline teacher responses/Gram solve excluded from deployment.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
