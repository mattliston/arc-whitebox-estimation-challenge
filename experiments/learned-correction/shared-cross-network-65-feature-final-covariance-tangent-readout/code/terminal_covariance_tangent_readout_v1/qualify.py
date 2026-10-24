"""Reuse qualified packets, reproducing immutable tangent features exactly."""
import json,resource,time
import numpy as np
from threadpoolctl import threadpool_info
from common import HERE,TCI,sha,read,seal_check,write
from feature import features
from fit_library import load_library
seal_check();assert all(t['num_threads']==1 for t in threadpool_info())
started=time.monotonic()
assert sha(HERE/'tangent.py')==sha(TCI/'tangent.py')
for name in ['unit','capture','qualification']:
    w=read(TCI/(name+'_watch.json'));assert w['completed'] and w['whole_group_retired'] and w['max_swap_growth_bytes']==0
assert read(TCI/'unit.json')['passed']
qual=read(TCI/'qualification/result.json');cap=read(TCI/'capture/result.json')
assert qual['completed'] and qual['passed'] and cap['completed'] and cap['repeat_all_captured_arrays_exact']
assert not read(TCI/'information/result.json')['information_gate_passed']
out=HERE/'qualification';out.mkdir(exist_ok=False)
library,equations=load_library();cert=library.certificate();records=[];first=None
old={r['rank']:r for r in qual['records']}
for row in cap['records']:
    assert sha(row['capture'])==row['capture_sha256']
    with np.load(row['capture'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    J,B=features(a)
    ref=old[row['rank']];assert sha(ref['path'])==ref['sha256']
    with np.load(ref['path'],allow_pickle=False) as z:
        assert np.array_equal(J,z['J'])
        assert np.array_equal(B,z['B'][:,[64]+list(range(64))])
        assert np.array_equal(a['prediction'],z['prediction'])
    if row['rank']==35:
        if first is None:first=(J.copy(),B.copy())
        else:assert np.array_equal(J,first[0]) and np.array_equal(B,first[1])
    path=out/f"call_{row['call']:02d}.npz"
    np.savez(path,features=J,hidden=a['H15'],sigma=a['sigma'],prediction=a['prediction'])
    records.append(dict(call=row['call'],rank=row['rank'],capture=str(path),capture_sha256=sha(path),parent_flops=row['parent_flops'],parent_multiplier=max(.1,row['parent_flops']/2**41),includes_initialization=row['call']==0,parent_all16_and_bill_exact=True,parent_hidden_and_sigma_exact=True,weights_sha256=row['weights_sha256'],numeric_cache_retired=True,truth_accessed=False,reused_parent_capture=row['capture'],reused_parent_capture_sha256=row['capture_sha256']))
    del a,J,B
write(out/'result.json',dict(completed=True,feature_repeat_exact=True,old_qualified_features_exact=True,records=records,fit_certificate=cert,fit_equation_checks=equations,source_seal_sha256=sha(HERE/'sources_sealed.json'),truth_accessed=False,new_model_calls=0,new_native_flops=0,old_oracle_failure_unchanged=True,elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(json.dumps(read(out/'result.json')),flush=True)
