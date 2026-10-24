"""Target-free original NumPy baselines; overlaps the fixed MPS fit safely."""
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from joint_prefix_final_numpy_v1 import run

HERE = Path(__file__).resolve().parent
FIT = HERE/'joint_prefix_final_fit_v1'
OUT = HERE/'joint_prefix_final_numpy_baselines_v1'
STORE = HERE.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
SOURCES = ('joint_prefix_final_numpy_v1.py','compact_fullpair_final_reference_v1.py',
           'radial_features.py','radial_gnn_numpy.py','compact_scalar_final_v1.py','compact_final_head_v1.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    fit = json.loads((FIT/'manifest.json').read_text())
    assert not fit['validation_targets_accessed']
    assert fit['protocol_sha256']==sha(HERE/'JOINT_PREFIX_FINAL_FIT_PROTOCOL_V1.md')
    P = dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
    Q = dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
    assert fit['original_P_sha256']==sha(HERE/'payload_compact_feed_fullpair_v1.npz')
    assert fit['original_Q_sha256']==sha(HERE/'payload_fullpair_final_head_v1.npz')
    threadpool_limits(limits=3)
    items = {item['role_rank']:item for item in json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']}
    OUT.mkdir()
    start = time.time()
    hashes = {name:sha(HERE/name) for name in SOURCES}
    records = []
    for rank in fit['validation_ranks']:
        item = items[rank]
        row = STORE/item['path']
        assert sha(row)==item['sha256']
        path = row.parent/'weights.npy'
        weights = np.load(path,mmap_mode='r')
        prediction = run(weights,P,Q)
        assert prediction.shape==(16,1024) and np.isfinite(prediction).all()
        out = OUT/f'rank_{rank:04d}.npy'
        np.save(out,prediction)
        record = dict(rank=rank,weight_sha256=sha(path),prediction_sha256=sha(out),
                      wall_seconds=time.time()-start,target_access=False)
        records.append(record)
        with (OUT/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        del weights,prediction
        gc.collect()
    assert hashes=={name:sha(HERE/name) for name in SOURCES}
    result = dict(completed=True,records=records,target_access=False,validation_targets_accessed=False,
                  original_P_sha256=fit['original_P_sha256'],original_Q_sha256=fit['original_Q_sha256'],
                  implementation_sha256=hashes,source_sha256=sha(__file__),
                  protocol_sha256=fit['protocol_sha256'],utc_finished=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.time()-start,peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  qualification='Original full dynamic NumPy bank predictions only, computed from weights with frozen P0/Q0. No targets, residuals, training, or validation score.')
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    main()
