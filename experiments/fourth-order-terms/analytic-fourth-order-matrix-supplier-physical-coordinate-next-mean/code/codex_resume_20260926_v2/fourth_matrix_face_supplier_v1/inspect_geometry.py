"""Read-only diagnosis of the supplier's pre-fit variance-domain failure."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
A=HERE.parents[1]/'astra_phase2_reexamination_v1'
out=HERE/'geometry.json';assert not out.exists()
f=np.load(A/'fourth_trace_oracle_inputs_r35_v1.npz',allow_pickle=False)
C=f['post12_covariance'];v=np.diag(C);bad=v<=0
alive=np.load(A/'rows_current_transport_shadow_v2/row_0035/alive.npy',allow_pickle=False)[12].astype(bool)
record=dict(width=len(v),variance_min=float(v.min()),variance_max=float(v.max()),
            nonpositive_count=int(bad.sum()),zero_count=int((v==0).sum()),negative_count=int((v<0).sum()),
            nonpositive_values=v[bad].tolist(),nonpositive_active_count=int((alive&bad).sum()),
            nonpositive_rows_max=float(np.max(np.abs(C[bad]))) if bad.any() else 0.,
            nonpositive_columns_max=float(np.max(np.abs(C[:,bad]))) if bad.any() else 0.,
            positive_variance_min=float(v[v>0].min()),truth_accessed=False)
with out.open('x') as o:json.dump(record,o,indent=2);o.write('\n')
print(json.dumps(record,indent=2),flush=True)
