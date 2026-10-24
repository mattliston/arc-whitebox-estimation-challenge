from pathlib import Path
import json, math
import numpy as np
from scipy.integrate import quad
from control import gaussian_mean, reference, coupled

rng = np.random.default_rng(392716)
errors = {}
qe = []
for mu, lv, d in ((0.,1.,.25),(-2.,.7,.12),(3.,.2,.5),(.4,2.,0.)):
    f = lambda z: float(gaussian_mean(mu+math.sqrt(lv)*z,d))*math.exp(-z*z/2)/math.sqrt(2*math.pi)
    got = quad(f,-12,12,epsabs=1e-12,epsrel=1e-12,points=[-mu/math.sqrt(lv)])[0]
    want = float(gaussian_mean(mu,lv+d))
    qe.append(abs(got-want))
assert max(qe)<2e-11
errors['centering_quadrature_max_abs'] = max(qe)
weights = rng.normal(size=(3,5,5))/math.sqrt(5)
ref, reports, mats = reference(weights,keep_matrices=True)
mineig = min(float(np.linalg.eigvalsh(row[key]).min()) for row in mats for key in ('Cpre','Cpost','noise'))
mind = min(float(np.linalg.eigvalsh(row['Cpre']-row['Lpre']@row['Lpre'].T).min()) for row in mats)
assert mineig > -1e-12 and mind > -1e-12
errors.update(min_covariance_eigenvalue=mineig,min_conditional_eigenvalue=mind)
X = rng.normal(size=(17,5))
out, layers = coupled(weights,ref,X,keep_layers=True)
assert np.max(np.abs(layers[0]['ordinary_residual']))<1e-14
assert np.max(np.abs(layers[0]['antithetic_residual']))<1e-14
errors['single_layer_abs'] = float(max(np.max(np.abs(layers[0][k])) for k in ('ordinary_residual','antithetic_residual')))

# Independently materialize every site's suffix and use the input map directly.
direct = np.broadcast_to(ref['m'][-1],(len(X),5)).copy()
direct += (X@mats[-1]['Lpre'].T)*ref['g'][-1]
anti = np.broadcast_to(ref['m'][-1],(len(X),5)).copy()
for i in range(3):
    y = X@mats[i]['Lpre'].T
    rp = gaussian_mean(ref['mu'][i]+y,ref['d'][i])-ref['m'][i]-ref['g'][i]*y
    ra = (gaussian_mean(ref['mu'][i]+y,ref['d'][i])+gaussian_mean(ref['mu'][i]-y,ref['d'][i]))/2-ref['m'][i]
    suffix = np.eye(5)
    for j in range(i+1,3): suffix = suffix@weights[j]*ref['g'][j][None,:]
    direct += rp@suffix; anti += ra@suffix
err = max(np.max(np.abs(out['plain']-out['ordinary_residual']-direct)),
          np.max(np.abs(out['antithetic']-out['antithetic_residual']-anti)))
assert err<1e-13
errors['explicit_site_sum_max_abs'] = float(err)
perms = [rng.permutation(5) for _ in range(4)]
wp = np.stack([weights[l][perms[l]][:,perms[l+1]] for l in range(3)])
rfp,_,_ = reference(wp)
op,_ = coupled(wp,rfp,X[:,perms[0]])
err = max(np.max(np.abs(op[k]-out[k][:,perms[-1]])) for k in out)
assert err<1e-13
errors['permutation_max_abs'] = float(err)
with (Path(__file__).parent/'unit.json').open('x') as f:
    json.dump(dict(passed=True,errors=errors,roundoff_reports=reports),f,indent=2,allow_nan=False)
print(json.dumps(dict(passed=True,errors=errors)),flush=True)
