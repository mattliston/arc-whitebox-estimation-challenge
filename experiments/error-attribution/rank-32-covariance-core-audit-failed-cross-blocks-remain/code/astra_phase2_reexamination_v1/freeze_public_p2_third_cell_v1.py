"""Actual bank prefix on the first metadata-selected mini training net."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'data_gate'))
import target_blind_mini_v2 as mini

OUT = HERE/'public_p2_third_cell_mini0057_v1.npz'
WEIGHTS = HERE/'public_p2_third_cell_mini0057_weights_v1.npy'
assert not OUT.exists() and not OUT.with_suffix('.json').exists() and not WEIGHTS.exists()
admission = json.loads((HERE/'public_p2_moments_metadata_v1.json').read_text())
assert admission['passed'] and admission['target_values_decoded'] == 0
row = admission['selected'][0]
assert row['ordinal'] == 57 and row['role'] == 'train'
entry = mini.AUTHORITY['splits']['mini']['shards'][row['shard_index']]
path = mini._attest_shard_pointer(mini.pinned_snapshot_path()/entry['path'], entry)
columns = ['mlp_id','mlp_name','mlp_seed','weights']
W = None
for index, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=1, columns=columns, use_threads=False)):
    assert batch.schema.names == columns
    if index == row['row_in_shard']:
        for key in columns[:3]:
            assert batch.column(batch.schema.get_field_index(key))[0].as_py() == row[key]
        W = mini._weights_view(batch).copy()
        break
assert W is not None and W.shape == (16,1024,1024)
payload = HERE/'payload_compact_feed_fullpair_v1.npz'
payload_hash = hashlib.sha256(payload.read_bytes()).hexdigest()
assert payload_hash == '166b3c67ec4d27793d433e38e15086b95d052426fdf435a79067cc07a2a4dd7e'
P = dict(np.load(payload))
state = np.zeros((1024,64), np.float32)
prefix, xs = [], []


class Complete(Exception):
    pass


def hook(layer, m, C, K21, aux, w):
    global state
    x = features(layer,m,C,K21,aux,w)
    delta,state = gnn_step(layer,x,aux['sigma'],w,state,P)
    prefix.append(m+delta)
    xs.append(x)
    if layer == 13:
        saved = dict(Cpre=aux['pre_cov'],K21=K21,mu=aux['alpha']*aux['sigma'],
                     sigma=aux['sigma'],alpha=aux['alpha'],pdf=aux['pdf'],gate=aux['gate'],
                     gaussian_mean=aux['gaussian_mean'],host=m,post_covariance=C,
                     post13_head_delta=delta,state=state,x=x,
                     prefix_means=np.stack(prefix),prefix_features=np.stack(xs))
        np.savez(OUT, **saved)
        np.save(WEIGHTS,W)
        result = dict(passed=True, ordinal=57, source_pre_layer=13, identity=row,
                      target_values_decoded=0, teacher_forcing=False, payload_sha256=payload_hash,
                      parent_source_sha256=hashlib.sha256(parent.PREDICT_SOURCE.encode()).hexdigest(),
                      prefix_mean_sha256=hashlib.sha256(saved['prefix_means'].tobytes()).hexdigest(),
                      weights_array_sha256=hashlib.sha256(W.tobytes()).hexdigest(),
                      weights_file_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                      output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
                      utc_finished=datetime.now(timezone.utc).isoformat(),
                      qualification='Unmodified parent full-pair prefix through13 on target-blind mini weights. Current learned state captured before mean feedback. mu reconstructed from alpha*sigma, not asserted bit-exact to pre-affine mu; all pair consumers use actual sigma/alpha.')
        OUT.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        raise Complete
    return delta


with threadpool_limits(limits=3):
    try:
        parent.predict(W,hook=hook,gate_floor=.01,cov32=True,metric='covariance')
    except Complete:
        pass
assert OUT.exists()
