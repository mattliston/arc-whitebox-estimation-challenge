"""Fixed terminal-only independent NumPy validation; no optimizer selection."""
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
P2 = HERE.parent
FIT = HERE/'joint_prefix_final_fit_v1'
OUT = HERE/'joint_prefix_final_judge_v1'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Preserve the existing independent judgment.'
    receipt = json.loads((FIT/'receipt.json').read_text())
    manifest = json.loads((FIT/'manifest.json').read_text())
    assert receipt['completed'] and receipt['updates']==240 and receipt['epochs']==6
    assert not receipt['validation_targets_accessed']
    assert manifest['protocol_sha256']==sha(HERE/'JOINT_PREFIX_FINAL_FIT_PROTOCOL_V1.md')
    payload = FIT/'terminal_payload.npz'
    assert receipt['terminal']['sha256']==sha(payload)
    assert receipt['validation_predictions_sha256']==sha(FIT/'terminal_validation_mps.npz')
    threadpool_limits(limits=3)
    with np.load(payload) as source:
        P = {k[2:]:source[k].copy() for k in source.files if k.startswith('P_')}
        Q = {k[2:]:source[k].copy() for k in source.files if k.startswith('Q_')}
    P0 = dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
    Q0 = dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
    for ns,old,new,trainkeys in (
        ('P',P0,P,('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','gain')),
        ('Q',Q0,Q,('w0','b0','w2','b2','w4','b4'))):
        assert set(old)==set(new),ns
        assert all(np.array_equal(old[k],new[k]) for k in old if k not in trainkeys),ns
    mps = np.load(FIT/'terminal_validation_mps.npz')
    role = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    items = {item['role_rank']:item for item in role}
    OUT.mkdir()
    start = time.time()
    records = []
    for rank in manifest['validation_ranks']:
        item = items[rank]
        row = STORE/item['path']
        assert sha(row)==item['sha256']
        weights = np.load(row.parent/'weights.npy',mmap_mode='r')
        base = run(weights,P0,Q0)
        current = run(weights,P,Q)
        assert np.isfinite(base).all() and np.isfinite(current).all()
        reference = mps[f'rank_{rank:04d}']
        difference = current-reference
        # First target access by this judge comes after its two independently
        # computed forward predictions, never inside either estimator.
        target_path = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        with np.load(target_path) as data:
            target = data['mc_post']
        base_layer = np.mean((base-target)**2,axis=1)
        layer = np.mean((current-target)**2,axis=1)
        record = dict(rank=rank,baseline_final_mse=float(base_layer[-1]),final_mse=float(layer[-1]),
                      ratio=float(layer[-1]/base_layer[-1]),win=bool(layer[-1]<base_layer[-1]),
                      baseline_layer_mse=base_layer.tolist(),layer_mse=layer.tolist(),
                      maximum_mps_difference=float(np.max(np.abs(difference))),
                      final_mps_difference_rms=float(np.sqrt(np.mean(difference[-1]**2))),
                      fidelity_passed=bool(np.max(np.abs(difference))<=3e-6),
                      target_sha256=sha(target_path),wall_seconds=time.time()-start)
        np.savez(OUT/f'rank_{rank:04d}.npz',baseline=base,current=current,mps=reference)
        records.append(record)
        with (OUT/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        del weights,base,current,reference,difference,target
        gc.collect()
    baseline = float(np.mean([r['baseline_final_mse'] for r in records]))
    final = float(np.mean([r['final_mse'] for r in records]))
    wins = sum(r['win'] for r in records)
    ratio = final/baseline
    fidelity = all(r['fidelity_passed'] for r in records)
    result = dict(passed=bool(fidelity and ratio<=.90 and wins>=24),fidelity_passed=fidelity,
                  baseline_final_mse=baseline,final_mse=final,pooled_raw_ratio=ratio,wins=wins,nets=len(records),
                  gates=dict(pooled_raw_ratio_max=.90,wins_min=24,maximum_mps_difference=3e-6),
                  records=records,terminal_payload_sha256=sha(payload),source_sha256=sha(__file__),
                  protocol_sha256=manifest['protocol_sha256'],utc_finished=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.time()-start,peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  qualification='Original and terminal jointly fitted full dynamic NumPy cards against existing 1e9 public-development means; one fixed 40-net validation endpoint. Not a 14-net grader panel, FLOP measurement, official score or pristine holdout.')
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    main()
