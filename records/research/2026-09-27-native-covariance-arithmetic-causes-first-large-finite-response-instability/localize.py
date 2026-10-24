from pathlib import Path
import hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'native_innovation_readout_response_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def load(name,mini):
    r=read(OLD/name/'complete.json');entry=next(x['packet'] for x in r['rows'] if x['mini']==mini)
    assert pin(Path(entry['path']))==entry
    with np.load(entry['path']) as z:return z['prediction'].copy()
for e in read(HERE/'seal.json')['sources']:assert pin(Path(e['path']))==e
old=read(OLD/'result.json');assert not old['passed'] and not old['probe_accuracy_scored']
rows=[];h=.125
for d in (0,7):
    for mini in (57,68):
        y0=load('zero',mini);p,m,ph,mh=[load(f'd{d}_{suffix}',mini) for suffix in ('plus','minus','halfplus','halfminus')]
        assert all(x[:9].tobytes()==y0[:9].tobytes() for x in (p,m,ph,mh))
        layers=[]
        for l in range(16):
            rh=(p[l]-m[l])/(2*h);rl=(ph[l]-mh[l])/h
            curvature_vector=p[l]+m[l]-2*y0[l]
            norm=lambda v:math.sqrt(math.fsum(float(x)**2 for x in v))
            denom=norm(rl);pair=norm(p[l]-m[l]);rms=h*denom/32
            discrepancy=norm(rh-rl)/denom if denom else 0.
            curvature=norm(curvature_vector)/pair if pair else 0.
            layers.append(dict(layer=l,secant_relative=discrepancy,response_rms=rms,
                curvature=curvature,nonlinear_remainder_rms=norm(curvature_vector)/32,
                rms_full_positive_change=norm(p[l]-y0[l])/32,
                rms_half_positive_change=norm(ph[l]-y0[l])/32,
                flagged=rms>=5e-7 and (discrepancy>.05 or curvature>.10)))
        prior=next(x for x in old['rows'] if x['mini']==mini and x['direction']==d)
        for key in ('secant_relative','response_rms','curvature'):
            assert math.isclose(layers[-1][key],prior[key],rel_tol=1e-13,abs_tol=1e-15)
        first=next((x['layer'] for x in layers if x['flagged']),None)
        rows.append(dict(direction=d,mini=mini,first_flagged_layer=first,layers=layers))
result=dict(rows=rows,previous_result=pin(OLD/'result.json'),first9_exact=True,
    final_metrics_replayed=True,final_targets_read=False,new_inference=False,
    coefficients_fitted=False,previous_failure_unchanged=True)
with (HERE/'localization.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(first_layers=[dict(direction=x['direction'],mini=x['mini'],layer=x['first_flagged_layer']) for x in rows],
    post9=[x['layers'][9] for x in rows])),flush=True)
