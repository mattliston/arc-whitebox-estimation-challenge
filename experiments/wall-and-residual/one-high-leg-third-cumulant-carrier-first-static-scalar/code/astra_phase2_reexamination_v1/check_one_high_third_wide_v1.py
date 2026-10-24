"""Fixed q4 current-host static information gate; stop on first failure."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step
from one_high_third_reference_v1 import compare

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True,choices=(35,100));args=ap.parse_args()
OUT=HERE/f'one_high_third_wide_r{args.rank}_v1.json';assert not OUT.exists()
assert json.loads((HERE/'one_high_third_small_v1.json').read_text())['passed']
assert json.loads((HERE/'one_high_third_cost_v1.json').read_text())['passed']
if args.rank==100:assert json.loads((HERE/'one_high_third_wide_r35_v1.json').read_text())['passed']
W=np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{args.rank:04d}'/'weights.npy')
P=dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
hstate=np.zeros((1024,P['head'].shape[1]),np.float32);records=[]
class FailedGate(Exception):pass


def hook(l,m,C,K21,aux,w):
    global hstate
    delta,hstate=gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,hstate,P)
    return delta


def probe(l,tape):
    if l not in (3,7,11,14):return
    t0=time.time();C=tape['C'].copy()
    ev,U=eigh((C+C.T)/2,subset_by_index=(1020,1023))
    result=compare(tape['p'].astype(np.float64),tape['lam'].astype(np.float64),tape['star'].astype(np.float64),U,W[l+1].astype(np.float64),
                   tape['b'].copy(),tape['Bpost'].copy())
    full,approx=result['full'],result['restored']
    relative=float(np.linalg.norm(approx-full)/np.linalg.norm(full))
    row=dict(rank=args.rank,layer=l,q=4,cp_columns=len(tape['lam']),
             covariance_trace_capture=float(ev.sum()/np.trace(C)),scalar_k3_relative_error=relative,
             full_k3_rms=float(np.sqrt(np.mean(full*full))),passed=relative<=.05,wall_s=time.time()-t0)
    records.append(row);print(json.dumps(row),flush=True)
    artifact=HERE/f'one_high_third_cell_r{args.rank}_l{l}_v1.npz';assert not artifact.exists()
    np.savez(artifact,full=full,retained=approx,U=U,covariance=C)
    if not row['passed']:raise FailedGate


src=parent.PREDICT_SOURCE
needle='        previous_alive = alive'
assert src.count(needle)==1
src=src.replace(needle,'        probe(l,locals())\n'+needle)
env=dict(parent.env,probe=probe);exec(src,env)
start=time.time()
with threadpool_limits(limits=3):
    try:env['predict'](W,hook=hook,gate_floor=.01)
    except FailedGate:pass
result=dict(rank=args.rank,records=records,passed=len(records)==4 and all(r['passed'] for r in records),
            stopped_on_first_failure=bool(records and not records[-1]['passed']),wall_s=time.time()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='Target-free static exact top4 covariance eigenspace information probe, CP and star compressed jointly, trace tensor left exact, repeated entries restored. Same current tensor as reference, not MC or estimator raw MSE. No recursive approximation, training or upload.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
