"""Verify saved scientific endpoint after the preserved final-logger error."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FIT = HERE/'joint_prefix_final_fit_v1'
OUT = HERE/'joint_prefix_final_terminal_integrity_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    manifest = json.loads((FIT/'manifest.json').read_text())
    receipt = json.loads((FIT/'receipt.json').read_text())
    failure = json.loads((FIT/'failure.json').read_text())
    lines = [json.loads(line) for line in (FIT/'progress.jsonl').read_text().splitlines()]
    assert sha(HERE/'train_joint_prefix_final_v1.py')==manifest['script_sha256']
    batches = [r for r in lines if r['stage']=='batch']
    epochs = [r for r in lines if r['stage']=='epoch_complete']
    terminal = [r for r in lines if r['stage']=='terminal_payload_written']
    validation = [r for r in lines if r['stage']=='validation_prediction_only']
    assert len(batches)==240 and [r['update'] for r in batches]==list(range(1,241))
    assert [r['epoch'] for r in epochs]==list(range(1,7))
    assert len(terminal)==1 and terminal[0]['sha256']==receipt['terminal']['sha256']
    assert [r['rank'] for r in validation]==manifest['validation_ranks']
    assert all(r['validation_targets']==0 for r in validation)
    assert terminal[0]['utc']<validation[0]['utc']<validation[-1]['utc']
    for epoch in range(1,7):
        members = [m['rank'] for r in batches if r['epoch']==epoch for m in r['metrics']]
        assert sorted(members)==sorted(manifest['train_ranks']) and len(members)==160
    assert all(np.isfinite(m[k]) for r in batches for m in r['metrics']
               for k in ('objective','final_mse','aux_mse'))
    payload = FIT/'terminal_payload.npz'
    assert sha(payload)==receipt['terminal']['sha256']
    with np.load(payload) as final, np.load(FIT/'epoch_06_audit_only.npz') as epoch6:
        assert set(final.files)==set(epoch6.files)
        assert all(np.array_equal(final[k],epoch6[k]) and np.isfinite(final[k]).all() for k in final.files)
        payload_keys = sorted(final.files)
    predictions = FIT/'terminal_validation_mps.npz'
    assert sha(predictions)==receipt['validation_predictions_sha256']
    with np.load(predictions) as pred:
        assert set(pred.files)=={f'rank_{rank:04d}' for rank in manifest['validation_ranks']}
        assert all(pred[k].shape==(16,1024) and np.isfinite(pred[k]).all() for k in pred.files)
    assert receipt['completed'] and receipt['epochs']==6 and receipt['updates']==240
    assert not receipt['validation_targets_accessed'] and not failure['validation_targets_accessed']
    assert failure['exception_type']=='TypeError'
    assert failure['message']=="dict() got multiple values for keyword argument 'wall_seconds'"
    assert "record(dict(stage='completed',**result))" in failure['traceback']
    assert 'value = dict(utc=utc(),wall_seconds=time.time()-started,**item)' in failure['traceback']
    assert validation[-1]['utc']<receipt['utc_finished']<=failure['utc_failed']
    provenance = HERE/'joint_prefix_final_provenance_v1/receipt.json'
    proof = json.loads(provenance.read_text())
    assert proof['passed']
    result = dict(passed=True,scientific_endpoint_complete=True,process_exit_clean=False,
                  error_is_post_serialization_status_logging=True,epochs_verified=6,updates_verified=240,
                  validation_predictions_verified=40,terminal_equals_epoch6_arrays=True,payload_keys=payload_keys,
                  terminal_payload_sha256=sha(payload),validation_predictions_sha256=sha(predictions),
                  completed_receipt_sha256=sha(FIT/'receipt.json'),preserved_failure_sha256=sha(FIT/'failure.json'),
                  provenance_receipt_sha256=sha(provenance),source_sha256=sha(__file__),
                  utc_finished=datetime.now(timezone.utc).isoformat(),target_arrays_accessed=False,
                  qualification='All requested optimization and target-free prediction artifacts were saved before the final duplicate-key status logger failed. Original failure.json, completed receipt, script and payloads remain untouched; no rerun, endpoint replacement, accuracy claim or clean-process-exit claim.')
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
