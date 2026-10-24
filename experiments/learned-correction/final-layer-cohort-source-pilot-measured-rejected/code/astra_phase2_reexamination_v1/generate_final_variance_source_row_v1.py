"""Input-only source row with an independent unchanged-parent replay."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from final_variance_source_reference_v1 import run
from final_variance_sources_v1 import NAMES

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);a=ap.parse_args()
assert a.rank==100 or 35<=a.rank<59
OUT=HERE/'rows_final_variance_source_probe_v1'/f'row_{a.rank:04d}'
if (OUT/'result.json').exists():
    old=json.loads((OUT/'result.json').read_text());assert old['passed']
    print(json.dumps(dict(rank=a.rank,reused=True)),flush=True);raise SystemExit(0)
assert not OUT.exists();OUT.mkdir(parents=True)
assert json.loads((HERE/'final_variance_sources_certificate_v1.json').read_text())['passed']
start=time.time()
with threadpool_limits(limits=2):
    instrument=run(a.rank,True)
    baseline=run(a.rank,False)
parity={k:bool(np.array_equal(instrument[k],baseline[k])) for k in ('prediction','h','host','sigma','delta','variance')}
finite=all(np.isfinite(v).all() for v in instrument.values())
pair=instrument['pair_relative'];cov=instrument['post_covariance_relative'];push=instrument['variance_sum_relative']
passed=all(parity.values()) and finite and max(pair,cov)<=1e-12 and push<=2e-6
record=dict(rank=a.rank,passed=passed,byte_parity=parity,pair_relative=pair,
            post_covariance_relative=cov,variance_sum_relative=push,feature_names=NAMES,
            wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            frozen_P_sha256=hashlib.sha256((HERE/'payload_compact_feed_fullpair_v1.npz').read_bytes()).hexdigest(),
            frozen_Q_sha256=hashlib.sha256((HERE/'payload_fullpair_final_head_v1.npz').read_bytes()).hexdigest(),
            qualification='No target file loaded. Frozen current dynamic prefix and Q, input-only final source channels, all-layer original-control parity. No truth feedback, retraining, mean error or score.')
np.savez(OUT/'data.npz',**instrument)
(OUT/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='feature_names'}),flush=True)
assert passed,record
