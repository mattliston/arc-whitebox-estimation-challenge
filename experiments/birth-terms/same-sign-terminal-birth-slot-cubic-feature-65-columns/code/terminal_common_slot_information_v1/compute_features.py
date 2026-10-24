"""Read saved target-free captures; seal exact repeated features."""
from common import HERE,sha,read,write,seal_check,guarded
seal_check();guarded('capture')
from feature import features
import numpy as np
capture=read(HERE/'capture/result.json')
assert capture['completed'] and capture['repeat_all_captured_arrays_exact']
rows=[];first=None
for r in capture['records']:
    assert sha(r['capture'])==r['capture_sha256']
    with np.load(r['capture']) as z:a={k:z[k] for k in z.files}
    J,B,proof=features(a)
    if r['rank']==35:
        if first is None:first=(J.tobytes(),B.tobytes())
        else:assert (J.tobytes(),B.tobytes())==first
    path=HERE/f'features_call{r["call"]:02d}.npz'
    np.savez(path,J=J,B=B,prediction=a['prediction'])
    rows.append(dict(**r,features=str(path),features_sha256=sha(path),**proof))
write(HERE/'features_sealed.json',dict(completed=True,rows=rows,repeat35_exact=True,truth_accessed=False,
    source_seal_sha256=sha(HERE/'sources_sealed.json'),features_unmetered_research=True))
print(__import__('json').dumps([{k:r[k] for k in ('rank','kernel_shape','collapsed_direction_scaled_error','k3_columns_rms')} for r in rows]))

