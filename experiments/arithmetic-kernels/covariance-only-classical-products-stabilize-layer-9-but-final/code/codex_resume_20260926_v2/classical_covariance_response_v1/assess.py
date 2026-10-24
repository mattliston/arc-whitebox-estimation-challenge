from pathlib import Path
import sys,json,hashlib,math
import numpy as np
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for e in read(HERE/'seal.json')['sources']:assert pin(e['path'])==e
def load(name):
    g=read(HERE/(name+'_guard.json'));assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    rows=read(HERE/name/'complete.json')['rows'];out={}
    for r in rows:
        e=r['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:out[r['mini']]={k:z[k].copy() for k in z.files}
    return out
assert sys.argv[1] in ('d0','all');directions=(0,) if sys.argv[1]=='d0' else (0,7)
load('disabled');zero=load('zero');rows=[]
def metrics(z,p,m,ph,mh):
    h=.125;rh=(p-m)/(2*h);rl=(ph-mh)/h
    squared=lambda x:math.fsum(float(v)**2 for v in x)
    secant=math.sqrt(squared(rh-rl)/max(squared(rl),1e-60))
    rms=math.sqrt(squared(h*rl)/len(z))
    curvature=math.sqrt(squared(p+m-2*z)/max(squared(p-m),1e-60))
    assert math.isclose(secant,float(np.linalg.norm(rh-rl)/max(np.linalg.norm(rl),1e-30)),rel_tol=1e-13,abs_tol=1e-15)
    gates=dict(secant=secant<=.05,resolvable=rms>=5e-7,curvature=curvature<=.10)
    return dict(secant_relative=secant,response_rms=rms,curvature=curvature,gates=gates,passed=all(gates.values()))
for direction in directions:
    data=[load(f'd{direction}_{suffix}') for suffix in ('plus','minus','halfplus','halfminus')]
    for mini in (57,68):
        layers={str(l):metrics(zero[mini]['prediction'][l],*[d[mini]['prediction'][l] for d in data]) for l in range(9,16)}
        cov=metrics(zero[mini]['Cpre9_diag'],*[d[mini]['Cpre9_diag'] for d in data])
        rows.append(dict(direction=direction,mini=mini,layers=layers,first_covariance_diagonal=cov,passed=layers['15']['passed']))
passed=all(r['passed'] for r in rows)
result=dict(passed=passed,rows=rows,directions=list(directions),final_targets_read=False,probe_accuracy_scored=False,
    coefficients_fitted=False,previous_qualification_unchanged=True,requires_direction7=passed and directions==(0,))
once(HERE/('direction0_result.json' if directions==(0,) else 'result.json'),result)
if directions==(0,) and not passed:once(HERE/'result.json',result)
print(json.dumps(result),flush=True)
