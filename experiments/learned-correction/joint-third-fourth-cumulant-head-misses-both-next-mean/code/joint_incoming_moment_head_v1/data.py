from common import *
import numpy as np
from math_core import coefficients
def load_cell(ordinal,layer):
    ir=read(OLD/f'inputs_mini{ordinal:04d}.json');tr=read(HERE/f'targets_mini{ordinal:04d}.json');assert ir['passed'] and tr['passed']
    i=next(r for r in ir['records'] if r['layer']==layer);t=next(r for r in tr['records'] if r['layer']==layer);check(i['input']);check(t['target'])
    with np.load(i['input']['path'],allow_pickle=False) as z:c={k:z[k] for k in ('F','R','S','Wnext','gnext')};s=z['sigma']
    cr=read(OLD/f'capture_mini{ordinal:04d}.json')
    with np.load(cr['capture']['path'],allow_pickle=False) as z:
        B,weight=coefficients(z[f'C_{layer:02d}'],s,z[f'alpha_{layer:02d}'],z[f'pdf_{layer:02d}'],z[f'gate_{layer:02d}'])
    c.update(B=B,weight=weight,target=np.load(t['target']['path'],allow_pickle=False),ordinal=ordinal,layer=layer)
    return c
