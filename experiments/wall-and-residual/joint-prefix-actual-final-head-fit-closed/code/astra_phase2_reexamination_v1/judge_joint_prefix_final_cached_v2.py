"""Same terminal-only judgment, reusing hash-verified target-free baselines."""
from datetime import datetime,timezone
import gc
import json
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from judge_joint_prefix_final_v1 import HERE,P2,FIT,STORE,sha,run

BASE = HERE/'joint_prefix_final_numpy_baselines_v1'
OUT = HERE/'joint_prefix_final_judge_cached_v2'


def main():
    assert not OUT.exists(), 'Preserve the existing independent judgment.'
    receipt = json.loads((FIT/'receipt.json').read_text())
    manifest = json.loads((FIT/'manifest.json').read_text())
    baseline_receipt = json.loads((BASE/'receipt.json').read_text())
    assert receipt['completed'] and receipt['updates']==240 and receipt['epochs']==6
    assert not receipt['validation_targets_accessed']
    assert baseline_receipt['completed'] and not baseline_receipt['target_access']
    assert manifest['protocol_sha256']==baseline_receipt['protocol_sha256']==sha(HERE/'JOINT_PREFIX_FINAL_FIT_PROTOCOL_V1.md')
    for filename,digest in baseline_receipt['implementation_sha256'].items():
        assert sha(HERE/filename)==digest,filename
    for ns,filename in (('P','payload_compact_feed_fullpair_v1.npz'),('Q','payload_fullpair_final_head_v1.npz')):
        assert baseline_receipt[f'original_{ns}_sha256']==manifest[f'original_{ns}_sha256']==sha(HERE/filename)
    baseline_rows = {r['rank']:r for r in baseline_receipt['records']}
    assert sorted(baseline_rows)==sorted(manifest['validation_ranks']) and len(baseline_rows)==40
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
        weight_file = row.parent/'weights.npy'
        assert sha(weight_file)==baseline_rows[rank]['weight_sha256']
        base_file = BASE/f'rank_{rank:04d}.npy'
        assert sha(base_file)==baseline_rows[rank]['prediction_sha256']
        base = np.load(base_file)
        weights = np.load(weight_file,mmap_mode='r')
        current = run(weights,P,Q)
        assert np.isfinite(base).all() and np.isfinite(current).all()
        reference = mps[f'rank_{rank:04d}']
        difference = current-reference
        # No target enters either estimator or baseline preparation.
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
                  baseline_preparation_receipt_sha256=sha(BASE/'receipt.json'),
                  protocol_sha256=manifest['protocol_sha256'],utc_finished=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.time()-start,peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  qualification='Same fixed 40-net terminal-only NumPy judgment as v1; original NumPy predictions were prepared target-free in parallel and hash-verified, not recomputed. Existing 1e9 public-development labels, not 14-net grader panel, metered bill, official score or pristine holdout.')
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    main()
