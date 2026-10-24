"""Compare completed observed platform runtimes, without score extrapolation."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
base = HERE / 'public_monitor_20261020T234528Z/eval_337421_raw.json'
current = max(HERE.glob('public_monitor_*/eval_337422_raw.json'), key=lambda p: p.stat().st_mtime)
a, b = (json.loads(p.read_text()) for p in (base, current))
old = {r['mlp_index']: r for r in a['evaluation']['results']['per_mlp']}
new = [r for r in b['evaluation']['results'].get('per_mlp', []) if r['state'] == 'completed']
assert len(old) == 100 and all(r['state'] == 'completed' for r in old.values())
assert new
pairs = []
for r in new:
    q = old[r['mlp_index']]
    assert r['mlp_name'] == q['mlp_name']
    t, u = r['telemetry'], q['telemetry']
    pairs.append(dict(mlp_index=r['mlp_index'], mlp_name=r['mlp_name'],
        predict_wall=t['participant_predict_wall_s'], prior_predict_wall=u['participant_predict_wall_s'],
        wall_pct=100 * (t['participant_predict_wall_s'] / u['participant_predict_wall_s'] - 1),
        residual=t['residual_wall_time_s'], prior_residual=u['residual_wall_time_s'],
        residual_delta=t['residual_wall_time_s'] - u['residual_wall_time_s'],
        bill_pct=100 * (t['flops_used'] / u['flops_used'] - 1)))


def stats(rows):
    out = {}
    for key in ('participant_predict_wall_s', 'residual_wall_time_s', 'participant_setup_wall_s'):
        values = [r['telemetry'].get(key) for r in rows]
        values = [x for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
        out[key] = dict(n_available=len(values), n_missing=len(rows) - len(values),
                        mean=statistics.mean(values) if values else None, max=max(values) if values else None)
    return out


record = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
    candidate_submission=337422, valid_reference_submission=337421,
    candidate_snapshot_utc=b['observed_utc'], candidate_state=b['evaluation']['state'],
    public_failed_count=b['evaluation']['progress']['mlps_failed'], reference_full100=stats(list(old.values())),
    candidate_completed_only=stats(new), paired_completed_only=dict(n=len(pairs),
        mean_wall_pct=statistics.mean(x['wall_pct'] for x in pairs),
        max_wall_pct=max(x['wall_pct'] for x in pairs),
        mean_residual_delta=statistics.mean(x['residual_delta'] for x in pairs),
        max_residual=max(x['residual'] for x in pairs),
        mean_bill_pct=statistics.mean(x['bill_pct'] for x in pairs)),
    worst_current_residual=sorted(pairs, key=lambda x: x['residual'], reverse=True)[:5],
    raw_paths=[str(base), str(current)], raw_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (base, current)],
    interpretation='Completed observed networks only; worker variation applies. No score extrapolation, hosted guarantee, reference mutation or fire action.')
path = HERE / ('RUNTIME_337422_VS_VALID337421_' + record['utc'].replace('-', '').replace(':', '') + '.json')
with path.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(receipt=str(path), **{k: record[k] for k in
    ('candidate_snapshot_utc', 'public_failed_count', 'candidate_completed_only', 'paired_completed_only')})))
