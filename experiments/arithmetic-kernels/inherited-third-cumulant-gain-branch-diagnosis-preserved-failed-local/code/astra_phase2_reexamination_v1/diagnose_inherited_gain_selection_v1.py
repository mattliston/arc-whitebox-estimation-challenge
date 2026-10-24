"""Target-free cause audit of the failed positive-step halving check.

Replay exactly the original three amplitudes, recording support decisions.
Then hold ONLY those original support decisions fixed for the same two
perturbations. This is a counterfactual diagnostic, not an estimator/fit.
"""
from pathlib import Path
import json
import resource
import time
import numpy as np
import inherited_distinct_gain_reference_v1 as reference

HERE=Path(__file__).resolve().parent
OUT=HERE/'inherited_gain_selection_diagnosis_v1.json'
ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
fidelity=json.loads((HERE/'inherited_distinct_gain_fidelity_v1.json').read_text())
saved=np.load(HERE/'inherited_distinct_gain_fidelity_v1.npz')
theta=np.array(fidelity['theta'])
original_select=reference.env['select']
traces={};preds={};start=time.time()


def one(name,control,frozen=None):
    trace=[]
    def select(score,floor):
        i=len(trace)
        idx,mask=original_select(score,floor)
        trace.append(dict(index=i,layer=i//2,kind='active' if i%2==0 else 'birth',
                          true_count=int(np.count_nonzero(score>=floor)),rounded_count=len(idx),
                          indices=idx.copy(),mask=mask.copy()))
        if frozen is not None:
            return frozen[i]['indices'].copy(),frozen[i]['mask'].copy()
        return idx,mask
    reference.env['select']=select
    try:
        prediction=reference.run(100,control)
    finally:
        reference.env['select']=original_select
    traces[name]=trace;preds[name]=prediction
    assert len(trace)==30
    print(json.dumps(dict(arm=name,wall_s=time.time()-start)),flush=True)


one('base',np.zeros(15))
one('finite',theta)
one('half',theta/2)
for name in ('base','finite','half'):
    assert np.array_equal(preds[name],saved[name+'_100']),name
one('fixed_finite',theta,traces['base'])
one('fixed_half',theta/2,traces['base'])
changes=[]
for name in ('finite','half'):
    for b,t in zip(traces['base'],traces[name]):
        different=int(np.count_nonzero(b['mask']!=t['mask']))
        if different:
            changes.append(dict(arm=name,layer=t['layer'],kind=t['kind'],
                                baseline_rounded=b['rounded_count'],new_rounded=t['rounded_count'],
                                baseline_true=b['true_count'],new_true=t['true_count'],
                                changed_memberships=different))
def halving(first,half):
    a=preds[first]-preds['base'];b=2*(preds[half]-preds['base'])
    return [float(np.linalg.norm(a[l]-b[l])/max(np.linalg.norm(a[l]),1e-30)) for l in range(16)]
result=dict(rank=100,original_three_replays_byte_exact=True,selection_changes=changes,
            dynamic_halving_by_layer=halving('finite','half'),
            fixed_selection_halving_by_layer=halving('fixed_finite','fixed_half'),
            wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='No targets, fitting, new amplitude, or deployment claim. Same original finite/half perturbations, with and without baseline support decisions. Explains the linear-response failure; does not reopen training under the failed preregistered gate.')
np.savez(ART,**preds,**{name+'_masks':np.stack([t['mask'] for t in trace]) for name,trace in traces.items()})
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
