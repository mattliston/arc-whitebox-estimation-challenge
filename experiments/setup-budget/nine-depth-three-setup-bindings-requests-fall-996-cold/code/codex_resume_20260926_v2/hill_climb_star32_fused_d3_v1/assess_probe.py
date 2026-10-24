"""Assess only frozen probe receipts and prediction parity; no target reads."""
from pathlib import Path
import io
import json
import math
import numpy as np
from run_probe import HERE, H2, TAG, checked, guard_check, jobs, once, pin

EVIDENCE = {}


def read(record):
    data = checked(record); EVIDENCE[record['path']] = record
    return data


def load(path):
    return json.loads(read(pin(path)))


def lines(record):
    return [json.loads(line) for line in read(record).decode().splitlines()]


def setup(seconds, log):
    assert math.isfinite(seconds) and 0 <= seconds <= 4.1
    assert [row[1] for row in log if row[0] == 'bound'] == [633]
    assert [row[1] for row in log if row[0] == 'lvfuse'] == [609]


def objects(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try: value, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError: continue
            if isinstance(value, dict): yield value


def boundary(data):
    assert [row['rank'] for row in data] == [35,100]
    for row in data:
        assert row['n_out'] == 16384 and row['nonfinite'] == 0
        assert row['flops'] == row['flops_ctx'] and row['flops'] > 0
        assert row['requests'] > 0 and math.isfinite(row['residual_s']) and row['residual_s'] >= 0
        assert math.isfinite(row['server_cpu_s']) and row['server_cpu_s'] > 0
    return data


def assess():
    protocol = load(HERE/'protocol.json')
    candidate, parent = protocol['candidate'], protocol['parent']
    assert read(candidate).startswith(read(parent))
    run = load(HERE/'probe_run.json'); attempt = json.loads(read(run['attempt']))
    assert run['completed'] and run['candidate'] == candidate and run['parent'] == parent
    assert run['attempt'] == pin(HERE/'probe_attempt.json')
    assert attempt['candidate'] == candidate and attempt['parent'] == parent
    assert attempt['protocol'] == pin(HERE/'protocol.json')
    for source in attempt['sources']: read(source)
    schedule = jobs(candidate['path'])
    assert attempt['jobs'] == [dict(name=n, seconds=s, command=c) for n,s,c in schedule]
    assert len(run['jobs']) == 3
    guards = []
    for (name, seconds, command), record in zip(schedule, run['jobs']):
        assert record == pin(HERE/(name+'_job.json'))
        job = json.loads(read(record))
        assert job['name'] == name and job['passed'] and not job['failures']
        assert job['candidate'] == candidate and job['command'] == command and job['seconds'] == seconds
        assert job['guard_exit_code'] == 0 and job['error'] is None
        assert job['guard'] == pin(HERE/(name+'_guard.json')) and job['log'] == pin(HERE/(name+'.log'))
        g = json.loads(read(job['guard'])); guard_check(g, command, seconds)
        read(job['log']); guards.append(g)
    native = attempt['native']
    nm = json.loads(read(native['manifest.json'])); nr = lines(native['rows.jsonl'])
    assert nm['candidate'] == parent['path'] and nm['sha256'] == parent['sha256'] and nm['ranks'] == [35,100]
    assert [row['rank'] for row in nr] == [35,100]
    worker = H2/(TAG+'_worker')
    wm = load(worker/'manifest.json'); wl = load(worker/'launcher.json'); wr = lines(pin(worker/'rows.jsonl'))
    assert wm['candidate'] == candidate['path'] and wm['sha256'] == candidate['sha256']
    assert wm['ranks'] == [35,100] and [row['rank'] for row in wr] == [35,100] and wm['server_instrumented']
    setup(wm['setup_s'], wm['pb_log'])
    assert wl['candidate_sha256'] == candidate['sha256'] and wl['client_exit'] == 0 and wl['server_retired']
    handles = wl['handles']
    assert handles['array_count_limit'] == 70000 and handles['peak_handles'] < 70000
    assert handles['stores'] and all(row['clears'] == 0 for row in handles['stores'])
    parity = []
    for i, rank in enumerate((35,100)):
        assert wr[i]['finite_outputs'] == 16384
        assert len(wr[i]['index_cache_entries']) == 3
        assert all(0 <= n <= cap for n,cap in zip(wr[i]['index_cache_entries'], (320,352,320)))
        meta = load(worker/f'pred_{rank}.json')
        assert meta['shape'] == [16,1024] and np.dtype(meta['dtype']) == np.dtype('float32')
        rpc = np.frombuffer(read(pin(worker/f'pred_{rank}.bin')), dtype=np.dtype(meta['dtype'])).reshape(16,1024)
        reference = np.load(io.BytesIO(read(native[f'pred_{rank}.npy'])), allow_pickle=False)
        assert reference.shape == (16,1024) and reference.dtype == np.dtype('float64')
        assert np.isfinite(rpc).all() and np.isfinite(reference).all()
        assert rpc.astype(np.float64).tobytes() == reference.tobytes(), 'All-layer promoted RPC bytes differ'
        assert wr[i]['flops'] == nr[i]['flops'], 'Worker FLOPs differ'
        parity.append(dict(rank=rank, all16_promoted_bytes_exact=True, exact_flops=nr[i]['flops']))
    resolution = json.loads(read(attempt['baseline_resolution']))
    assert resolution['passed'] and resolution['candidate'] == parent
    assert resolution['fresh_candidate'] == attempt['baseline']
    base = attempt['baseline']
    bg = json.loads(read(base['guard'])); bl = json.loads(read(base['launcher']))
    assert bl['candidate'] == parent['path'] and bl['candidate_sha256'] == parent['sha256']
    assert bl['client_exit'] == 0 and bl['server_retired']
    base_tag = Path(base['census']['path']).name.removesuffix('_census.json')
    from run_probe import LIGHT
    guard_check(bg, [str(LIGHT), '-B', str(H2/'run_boundary_census.py'), parent['path'], '35,100', base_tag], 240)
    baseline = boundary(json.loads(read(base['census'])))
    current = boundary(load(H2/(TAG+'_boundary_census.json')))
    launcher = load(H2/(TAG+'_boundary_launcher.json'))
    assert launcher['candidate'] == candidate['path'] and launcher['candidate_sha256'] == candidate['sha256']
    assert launcher['client_exit'] == 0 and launcher['server_retired']
    setup_rows = [r for r in objects(read(pin(HERE/'boundary.log')).decode()) if 'setup_wall_s' in r]
    assert len(setup_rows) == 1
    setup(setup_rows[0]['setup_wall_s'], setup_rows[0]['setup_log'])
    comparisons = []
    for i,(actual,reference) in enumerate(zip(current,baseline)):
        assert actual['flops'] == reference['flops'] == nr[i]['flops'], 'Boundary FLOPs differ'
        comparisons.append(dict(rank=actual['rank'], requests=actual['requests'], parent_requests=reference['requests'],
            request_delta=actual['requests']-reference['requests'], residual_s=actual['residual_s'],
            parent_residual_s=reference['residual_s'], residual_delta_s=actual['residual_s']-reference['residual_s'],
            free_trip_delta=actual['free_trips']-reference['free_trips'],
            server_cpu_ratio=actual['server_cpu_s']/reference['server_cpu_s']))
    stress = load(H2/(TAG+'_setup')/'summary.json')
    assert stress['candidate_sha256'] == candidate['sha256']
    assert [r['repeat'] for r in stress['records']] == [0,1,2]
    for row in stress['records']:
        assert row['candidate'] == candidate['path'] and row['sha256'] == candidate['sha256'] and row['load_processes'] == 2
        setup(row['setup_s'], row['pb_log'])
    for record in list(EVIDENCE.values()): read(record)
    return dict(passed=True, candidate=candidate, parent=parent, parity=parity,
        boundary=comparisons, expected_steady_arithmetic_request_reduction=870,
        setup=dict(bound=633, fused=609, worker_s=wm['setup_s'], boundary_s=setup_rows[0]['setup_wall_s'],
                   stress_s=[r['setup_s'] for r in stress['records']]),
        max_probe_rss_bytes=max(g['max_rss_bytes'] for g in guards), zero_added_swap=True,
        worker_networks=2, measured_peak_handles=handles['peak_handles'],
        note='Parity/resource/setup qualification only. Boundary deltas compare two35/100 processes; timing is descriptive. No score claim, worker20 qualification or submission admission.',
        evidence=list(EVIDENCE.values()), assessor=pin(__file__))


def main():
    target = HERE/'probe_assessment.json'
    assert not target.exists()
    try:
        result = assess()
    except Exception as exc:
        once(target, dict(passed=False, error=repr(exc), evidence=list(EVIDENCE.values()), assessor=pin(__file__)))
        raise
    once(target, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'evidence'}), flush=True)


if __name__ == '__main__':
    main()
