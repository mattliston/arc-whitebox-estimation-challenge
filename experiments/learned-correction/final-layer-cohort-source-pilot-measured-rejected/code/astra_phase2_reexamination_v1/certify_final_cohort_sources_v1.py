"""Small target-free algebra certificate before any wide row generation."""
from pathlib import Path
import json
import numpy as np
from final_cohort_sources_v1 import split

HERE = Path(__file__).resolve().parent
OUT = HERE/'final_cohort_sources_certificate_v1.json'
assert not OUT.exists()
rng = np.random.default_rng(930065)
records = []
for n in (8, 13):
    counts = (3, 5, 7)
    factors, weights = [], []
    for count in counts:
        u = rng.normal(size=(n, count))*.2
        d = rng.normal(size=(n, count))*.1
        coeff = rng.normal(size=count)
        factors.extend((u+d/2, u-d/2))
        weights.extend((coeff, -coeff))
    pp, lam = np.concatenate(factors, axis=1), np.concatenate(weights)
    direct = np.sum(lam*pp**3, axis=1)
    decomposed = split(pp, lam, counts).sum(axis=1)
    rel = float(np.linalg.norm(direct-decomposed)/np.linalg.norm(direct))
    records.append(dict(width=n, relative_error=rel, passed=rel<=1e-12))
result = dict(passed=all(r['passed'] for r in records), records=records)
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
assert result['passed']
