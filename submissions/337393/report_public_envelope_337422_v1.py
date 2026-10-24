"""Saved public100 runtime envelope; no estimate of future hosted safety."""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
RAW = HERE / 'public_monitor_20261021T002129Z/eval_337422_raw.json'
EXPECTED = '5345808663a5b6794d55f2b16d7753106736101bebd13be2898f4a2e0da591d7'
assert hashlib.sha256(RAW.read_bytes()).hexdigest() == EXPECTED
data = json.loads(RAW.read_text())
e = data['evaluation']
rows = e['results']['per_mlp']
setups = e['results']['setup_executions']
assert e['state'] == 'GRADED' and len(rows) == 100
assert all(r['state'] == 'completed' and r.get('error_code') in (None, 'SUCCESS') for r in rows)
assert e['progress']['mlps_failed'] == 0
assert len(setups) == 5 and all(s['error_code'] == 'SUCCESS' for s in setups)
keys = ('participant_predict_wall_s', 'flopscope_server_kernel_wall_s',
        'flopscope_server_overhead_wall_s', 'residual_wall_time_s',
        'participant_roundtrip_wall_s', 'flopscope_server_comms_wall_s')
summary = {}
for key in keys:
    values = [r['telemetry'][key] for r in rows]
    assert all(isinstance(v, (float, int)) for v in values)
    largest = max(rows, key=lambda r: r['telemetry'][key])
    summary[key] = dict(n=len(values), mean=statistics.mean(values),
                        max=max(values), max_mlp_index=largest['mlp_index'],
                        max_mlp_name=largest['mlp_name'])

def network(r):
    t = r['telemetry']
    return dict(mlp_index=r['mlp_index'], mlp_name=r['mlp_name'],
                telemetry={key: t[key] for key in keys},
                remaining_predict_seconds=120.0-t['participant_predict_wall_s'],
                remaining_residual_seconds=0.4-t['residual_wall_time_s'],
                additional_predict_fraction_before_limit=120.0/t['participant_predict_wall_s']-1,
                additional_residual_fraction_before_limit=0.4/t['residual_wall_time_s']-1)

record = dict(
    utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
    submission=337422, snapshot_utc=data['observed_utc'], completed_at=e['completed_at'],
    raw_path=str(RAW), raw_sha256=EXPECTED, state=e['state'], failed=0,
    complete_networks=100, setup_successful=5, summary=summary,
    setup=dict(mean_seconds=statistics.mean(s['wall_s'] for s in setups),
               max_seconds=max(s['wall_s'] for s in setups),
               remaining_seconds_at_max=5.0-max(s['wall_s'] for s in setups)),
    observed_worst_wall=network(max(rows, key=lambda r:r['telemetry']['participant_predict_wall_s'])),
    observed_worst_residual=network(max(rows, key=lambda r:r['telemetry']['residual_wall_time_s'])),
    caps=dict(predict_seconds=120.0, residual_seconds=0.4, setup_seconds=5.0,
              local_authoritative_source='lead/runs/astra_crunch_runtime_v1/failure336631_v1/whestbench_0.16.1/whestbench/runner.py:62-63; scoring.py:68-71'),
    interpretation='Observed full100 margins only. Different worst networks are kept separate; no permission to spend the margin, predicted kernel-runtime transfer, or guarantee for later hosted runs. Edge8 has original Mac32 and independent fresh4 evidence; no separate quiet production-boundary headroom result was found in the current artifacts.')
out = HERE / 'PUBLIC_337422_FULL100_RUNTIME_ENVELOPE_v1.json'
with out.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(receipt=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    summary=summary,setup=record['setup'],worst_wall=record['observed_worst_wall'],
    worst_residual=record['observed_worst_residual'])))
