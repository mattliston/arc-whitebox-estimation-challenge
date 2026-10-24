"""Passive prefix capture and fixed target-free Gaussian tree source."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from tree_v2 import source, diagonal, post_q22

HERE=Path(__file__).resolve().parent
P2=HERE.parents[1];ASTRA=P2/'astra_phase2_reexamination_v1'
sys.path.insert(0,str(ASTRA))
import radial_compact_feed_fullpair_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return h.hexdigest()


seal=json.loads((HERE/'seal_v2.json').read_text())
for path,digest in seal['files'].items():assert sha(Path(path))==digest,path
unit=json.loads((HERE/'units.json').read_text());assert unit['passed']
assert unit['tree_sha256']==sha(HERE/'tree_v2.py')
guard=json.loads((HERE/'units_v2_guard.json').read_text())
assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
out=HERE/'capture.json';assert not out.exists() and not out.with_suffix('.npz').exists()
start=time.monotonic()
folder=ASTRA/'rows_current_transport_shadow_v2/row_0035'
W=np.load(folder/'weights.npy',allow_pickle=False)
oldx=np.load(folder/'x.npy',mmap_mode='r');olds=np.load(folder/'sigma.npy',mmap_mode='r')
P=dict(np.load(ASTRA/'payload_compact_feed_fullpair_v1.npz',allow_pickle=False))
f=dict(np.load(ASTRA/'fourth_trace_oracle_inputs_r35_v1.npz',allow_pickle=False))
state=np.zeros((1024,64),np.float32);checks=[];saved={}


class Complete(Exception):pass


def hook(layer,m,C,K21,aux,w):
    global state
    x=features(layer,m,C,K21,aux,w)
    checks.append(bool(np.array_equal(x,oldx[layer]) and np.array_equal(aux['sigma'].astype(np.float32),olds[layer])))
    assert checks[-1],layer
    if layer==12:
        raw=aux['pre_cov'];saved['Cpre12']=(raw+raw.T)/2
        saved['g12']=aux['gate'].copy()
        saved['d12']=(aux['pdf']/aux['sigma']).copy()
        saved['e12']=(-aux['alpha']*aux['pdf']/aux['sigma']**2).copy()
        saved['pre12_asymmetry']=np.asarray(np.max(np.abs(raw-raw.T)))
    delta,state=gnn_step(layer,x,aux['sigma'],w,state,P)
    return delta


def post(layer,values):
    if layer!=12:return
    assert len(checks)==13 and all(checks)
    assert np.array_equal(values['C'],f['post12_covariance'])
    assert float(values['radial_new'])==float(f['post12_radial'])
    Cpost=(values['C']+values['C'].T)/2;w=W[13].astype(np.float64)
    args=tuple(saved[k] for k in ('Cpre12','g12','d12','e12'))
    delta4,delta31,info,tree4,tree31=source(*args,w,Cpost)
    diag_error=float(np.max(np.abs(diagonal(*args,w)-tree4)))
    assert diag_error<=1e-12*max(1.,float(np.max(np.abs(tree4)))),diag_error
    radial22=info['beta']*(np.outer(np.diag(Cpost),np.diag(Cpost))+2*Cpost*Cpost)/3
    trace_error=float(abs(np.sum(post_q22(*args)-radial22)))
    assert trace_error<=1e-10*max(1.,abs(info['trace'])),trace_error
    assert np.array_equal(w,f['W13'].astype(np.float64))
    np.savez(out.with_suffix('.npz'),**saved,Cpost12=Cpost,W13=w,
             delta4=delta4,delta31=delta31,tree4=tree4,tree31=tree31)
    record=dict(passed=True,prefix_feature_sigma_byte_parity=checks,
                frozen_post12_covariance_byte_parity=True,source=info,
                marginal_diagonal_error=diag_error,birth_trace_cancellation_error=trace_error,
                pre12_asymmetry=float(saved['pre12_asymmetry']),truth_accessed=False,
                output_sha256=sha(out.with_suffix('.npz')),source_seal_sha256=sha(HERE/'seal_v2.json'),
                wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                utc_finished=datetime.now(timezone.utc).isoformat())
    with out.open('x') as o:json.dump(record,o,indent=2);o.write('\n')
    print(json.dumps(record,indent=2),flush=True)
    raise Complete


code=parent.PREDICT_SOURCE
old='        radial = radial_new';assert code.count(old)==1
code=code.replace(old,'        post(l,locals())\n'+old)
env=dict(parent.env,post=post);exec(code,env)
try:env['predict'](W,hook=hook,gate_floor=.01)
except Complete:pass
assert out.exists()
