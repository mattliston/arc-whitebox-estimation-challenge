"""Read-only current pre14 response; independently reproduce frozen pre13 bytes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step

HERE=Path(__file__).resolve().parent
OUT=HERE/'public_p2_third_next_response_v1.npz'
assert not OUT.exists() and not OUT.with_suffix('.json').exists()
frozen=HERE/'public_p2_third_cell_mini0057_v1.npz'
fr=json.loads(frozen.with_suffix('.json').read_text())
assert hashlib.sha256(frozen.read_bytes()).hexdigest()==fr['output_sha256']
f=dict(np.load(frozen));W=np.load(HERE/'public_p2_third_cell_mini0057_weights_v1.npy')
P=dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
state=np.zeros((1024,64),np.float32)
prefix=[];xs=[];checks={}


class Complete(Exception):pass


def hook(layer,m,C,K21,aux,w):
    global state
    x=features(layer,m,C,K21,aux,w)
    delta,state=gnn_step(layer,x,aux['sigma'],w,state,P)
    if layer==14:
        assert checks and all(checks.values()),checks
        np.savez(OUT,response14=aux['pdf']/(2*aux['sigma']),sigma14=aux['sigma'],alpha14=aux['alpha'])
        record=dict(passed=True,original_cell_byte_parity=checks,target_values_decoded=0,
                    frozen_input_sha256=fr['output_sha256'],output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
                    utc_finished=datetime.now(timezone.utc).isoformat())
        OUT.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
        raise Complete
    prefix.append(m+delta);xs.append(x)
    if layer==13:
        current=dict(Cpre=aux['pre_cov'],K21=K21,mu=aux['alpha']*aux['sigma'],sigma=aux['sigma'],
                     alpha=aux['alpha'],pdf=aux['pdf'],gate=aux['gate'],gaussian_mean=aux['gaussian_mean'],
                     host=m,post_covariance=C,post13_head_delta=delta,state=state,x=x,
                     prefix_means=np.stack(prefix),prefix_features=np.stack(xs))
        checks.update({key:bool(np.array_equal(value,f[key])) for key,value in current.items()})
        assert all(checks.values()),checks
    return delta


with threadpool_limits(limits=3):
    try:parent.predict(W,hook=hook,gate_floor=.01,cov32=True,metric='covariance')
    except Complete:pass
assert OUT.exists()
