"""Single saved-v61j falsifier; no SDK, model, or fitted parameter."""
import ast
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

D = Path(__file__).resolve().parent
A = D.parent
source = A/'setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
capture = A/'PRICE_CAPTURE_RANK16_v1.npz'
assert hashlib.sha256(capture.read_bytes()).hexdigest() == '563744cd09dcaeb68243c986eaf8f4b05082c0b914c5acb50791d6e78806fbea'
tables = {}
for node in ast.parse(source.read_text()).body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ('_GN_RC', '_GN_KC'):
                tables[target.id] = ast.literal_eval(node.value)
z = np.load(capture)
started = time.monotonic()
rows = []
for layer in (13,14):
    m = z[f'L{layer}_host'].astype(np.float64)
    dm = np.maximum(z[f'L{layer}_delta'].astype(np.float64), -m)
    C0 = z[f'L{layer}_cov_after_edge'].astype(np.float64)
    rc = tables['_GN_RC'].get(layer, 0) - tables['_GN_KC'].get(layer, 0)
    diag0 = np.diag(C0).copy()
    md = m*dm
    diag = ((diag0-md)-md)-dm*dm
    diag = diag-rc*(diag0-diag)
    m32, dm32, c32 = m.astype(np.float32), dm.astype(np.float32), C0.astype(np.float32)
    md32 = np.outer(m32,dm32)
    C = ((c32-md32)-md32.T)-np.outer(dm32,dm32)
    C = C-np.float32(rc)*(c32-C)
    C = C.astype(np.float64)
    np.fill_diagonal(C,diag)
    # QR is only 1024x2; projection costs 4.2MF per matrix, no full eigensolve.
    Q, _ = np.linalg.qr(np.column_stack((m,dm)))
    before = (Q.T@C0)@Q
    after = (Q.T@C)@Q
    eb = np.linalg.eigvalsh((before+before.T)/2)
    ea = np.linalg.eigvalsh((after+after.T)/2)
    rows.append(dict(layer=layer, span='host mean and effective learned mean correction',
        rc_minus_kc=rc, before_eigenvalues=eb.tolist(), after_eigenvalues=ea.tolist(),
        negative_ritz_mass=float(-np.minimum(ea,0).sum()), negative_diagonal_count=int((diag<0).sum()),
        negative_diagonal_mass=float(-np.minimum(diag,0).sum()), covariance_trace=float(diag.sum()),
        tiny_diagonal_count=int((diag<1e-12).sum()),
        mean_correction_rms=float(np.sqrt(np.mean(dm*dm)))))
result = dict(status='CLOSED_SPAN_HAS_NO_NEGATIVE_CURVATURE' if all(r['negative_ritz_mass']==0 for r in rows) else 'NEGATIVE_CURVATURE_REQUIRES_TRUTH_PROXY',
    rank=16, parent_capture='v61j, same S56 physics as current95c; kernel lineage differs',
    current_best_adjusted_score=1.4468565569762618e-9, rows=rows,
    approximate_added_bill_pct_per_layer=.003, native_gain_banked=False,
    elapsed_s=time.monotonic()-started, maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (D/'PSD_SPAN_RESULT_v1.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
