"""The passing fixed source sum with actual positive-resummed path births."""
import json
from pathlib import Path
import numpy as np
import cumulative_backward_source_v1 as base
from positive_path_resummation_v1 import factors,positive_factor,birth_difference_slices


def certificate():
    original=base.certificate()
    extra=json.loads(Path(__file__).resolve().with_name('positive_path_resummation_certificate_v1.json').read_text())
    assert original['passed'] and extra['passed']
    return dict(passed=True,original_multistep=original,positive_path=extra)


def source_slices(K21,K22,eta,births,**kwargs):
    args=dict(kwargs);args['e']=np.zeros_like(args['e'])
    result=base.source_slices(**args) # unchanged defect terms and triangle
    B,E=factors(kwargs['C'],kwargs['g'],kwargs['d'],eta,K21,K22,kwargs['radial'])
    Z,_=positive_factor(B,E)
    t=kwargs['d']*(1+kwargs['radial']/3)*births
    return result+birth_difference_slices(B,Z,t,kwargs['H'])
