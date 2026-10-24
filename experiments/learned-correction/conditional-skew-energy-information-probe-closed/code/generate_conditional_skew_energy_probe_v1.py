"""Small spectral regression statistics of saved honest teacher pre-states."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

here=Path(__file__).resolve().parent
out=here/'conditional_skew_energy_probe_rows_v1.json'
folder=here/'rows_conditional_skew_energy_probe_v1'
assert not out.exists() and not folder.exists()
assert json.loads((here/'learned_joint_graph_probe_rows_v1.json').read_text())['n']==24
folder.mkdir()
start=time.time();records=[]
with threadpool_limits(limits=4):
    for rank in range(35,59):
        parent=here/'rows_learned_joint_graph_probe_v1'/f'row_{rank:04d}'
        assert json.loads((parent/'result.json').read_text())['parent_byte_exact']
        raw=np.load(here/'rows_compact_feed_fullpair_v1'/f'row_{rank:04d}'/'raw20.npy')
        with np.load(parent/'data.npz') as z:
            result=[];captured=[]
            for j,l in enumerate(range(10,16)):
                s=z['sigma'][j].astype(np.float64)
                R=z['r'][j].astype(np.float64);np.fill_diagonal(R,1.)
                C=R*np.outer(s,s);C=(C+C.T)/2
                T=z['t'][j].astype(np.float64)
                skew=raw[l,:,10].astype(np.float64);np.fill_diagonal(T,skew)
                values,U=eigh(C,subset_by_index=(1016,1023),check_finite=False)
                assert values.min()>0
                b=T@(s[:,None]*U/np.sqrt(values)[None,:])
                c=U*np.sqrt(values)[None,:]/s[:,None]
                result.append(np.stack(((b*b).sum(1),(b*c).sum(1),(c*c).sum(1),
                                        ((b-skew[:,None]*c)**2).sum(1)),axis=1).astype(np.float32))
                captured.append(float(values.sum()/np.trace(C)))
            feat=np.stack(result)
            assert feat.shape==(6,1024,4) and np.isfinite(feat).all()
            np.save(folder/f'rank_{rank:04d}.npy',feat)
        records.append(dict(rank=rank,parent_byte_exact=True,top8_variance_fraction=captured,
                            feature_mean=feat.mean((0,1)).tolist()))
        print(json.dumps(dict(rank=rank,wall_s=time.time()-start)),flush=True)
out.write_text(json.dumps(dict(n=24,records=records,wall_s=time.time()-start,
                              provenance='Saved honest pre-state only; no current-target feedback in features.'),indent=2)+'\n')
print('CONDITIONAL_SKEW_ENERGY_ROWS_COMPLETE',flush=True)
