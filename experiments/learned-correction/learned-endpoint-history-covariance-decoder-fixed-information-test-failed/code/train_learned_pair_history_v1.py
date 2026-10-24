"""Frozen learned endpoint histories; exact old objective and fixed fit."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
receipt = json.loads((HERE/'learned_pair_history_rows_v1.json').read_text())
assert receipt['n'] == 8 and receipt['passed']
path = HERE/'train_covariance_innovation_probe_v1.py'
source = path.read_text()
edits = (
    ("out=here/'covariance_innovation_probe_verdict_v1.json'", "out=here/'learned_pair_history_verdict_v1.json'"),
    ("payload=here/'covariance_innovation_probe_model_v1.npz'", "payload=here/'learned_pair_history_model_v1.npz'"),
    ("assert json.loads((here/'covariance_innovation_probe_rows_v1.json').read_text())['n']==8",
     "assert json.loads((here/'learned_pair_history_rows_v1.json').read_text())['passed']\nassert 'threadpoolctl' not in __import__('sys').modules"),
    ("here/'rows_covariance_innovation_probe_v1'", "here/'rows_covariance_innovation_probe_v2'"),
    ("        records[rank]={k:torch.from_numpy(z[k].copy()) for k in z.files}",
     "        records[rank]={k:torch.from_numpy(z[k].copy()) for k in z.files}\n    hp=here/'rows_learned_pair_history_v1'/f'row_{rank:04d}/history.npy'\n    hr=json.loads(hp.with_name('result.json').read_text())\n    assert hashlib.sha256(hp.read_bytes()).hexdigest()==hr['history_sha256']\n    h=np.load(hp)\n    records[rank]['basis']=torch.from_numpy(np.concatenate((h,np.ones((*h.shape[:2],1),np.float32)),axis=-1))"),
    ('.clamp_min(1e-8)', '.clamp_min(1e-6)'),
    ('torch.zeros((3,14,14),device=dev)', 'torch.zeros((3,65,65),device=dev)'),
    ("qualification='Oracle-state supervised covariance source pilot; two held-out public networks, stored MC covariance labels. Not a deployable or panel score; no independent covariance-replica noise correction.'",
     "decoder_features=65, frozen_P0=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n    original_trainer_sha256=ORIGINAL_TRAINER_SHA,\n    qualification='Frozen learned two-endpoint covariance decoder, coherent8M teacher state;6train/2heldout, fixed100epochs, no new labels or state writes. Not a deployable/panel score; shared MC covariance noise not independently corrected.'"),
)
for before, after in edits:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
sha = hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(dict(original_trainer_sha256=sha, features=65, train=6, valid=2,
                      epochs=100, layers=list(range(8, 15)),
                      normalization='training RMS only,1e-6 floor, constant appended',
                      gate='<=.75 zero-decoder covariance AND next-mean-response MSE on EACH held-out net')), flush=True)
exec(compile(source, str(path), 'exec'), dict(__file__=__file__, __name__='__main__',
                                           hashlib=hashlib, ORIGINAL_TRAINER_SHA=sha))
