"""Root-only receipt assessment. Reads saved predictions, never targets/models."""
from pathlib import Path
import io
import json
import math
import numpy as np
from run_probe import (HERE, H2, LIGHT, TAG, checked, guard_check, native_guard_check,
                       jobs, once, pin)

EVIDENCE = {}


def read(record):
    data = checked(record)
    EVIDENCE[record['path']] = record
    return data


def load(path):
    return json.loads(read(pin(path)))


def lines(record):
    return [json.loads(line) for line in read(record).decode().splitlines()]


def setup(seconds, log):
    assert math.isfinite(seconds) and 0 <= seconds <= 4.1
    assert [r[1] for r in log if r[0] == 'bound'] == [651]
    assert [r[1] for r in log if r[0] == 'lvfuse'] == [627]


def objects(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                value, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def boundary(data):
    assert [r['rank'] for r in data] == [35,100]
    for r in data:
        assert r['n_out'] == 16384 and r['nonfinite'] == 0
        assert r['flops'] == r['flops_ctx'] and r['flops'] > 0
        assert r['requests'] > 0 and math.isfinite(r['residual_s']) and r['residual_s'] >= 0
        assert math.isfinite(r['server_cpu_s']) and r['server_cpu_s'] > 0
    return data


def prediction(record):
    array = np.load(io.BytesIO(read(record)), allow_pickle=False)
    assert array.shape == (16,1024) and array.dtype == np.dtype('float64')
    assert np.isfinite(array).all()
    return array


def assess():
    build = load(HERE/'build.json')
    candidate, parent = build['candidate'], build['parent']
    assert read(candidate) == read(parent) + read(build['fragment'])
    for record in [build['builder'], build['protocol'], *build['sources']]:
        read(record)
    run = load(HERE/'probe_run.json')
    attempt = json.loads(read(run['attempt']))
    assert run['completed'] and run['candidate'] == candidate and run['parent'] == parent
    assert run['attempt'] == pin(HERE/'probe_attempt.json')
    assert attempt['candidate'] == candidate and attempt['parent'] == parent
    assert attempt['build'] == pin(HERE/'build.json')
    for record in attempt['sources']:
        read(record)
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
        guard = json.loads(read(job['guard']))
        guard_check(guard, command, seconds)
        read(job['log'])
        guards.append(guard)

    # Recheck root's completed native gate without running it or replacing it.
    native_receipt = json.loads(read(attempt['native_assessment']))
    assert run['native_admission'] == attempt['native_assessment'] == pin(HERE/'native2_assessment.json')
    assert native_receipt['passed'] and native_receipt['candidate'] == candidate
    assert all(native_receipt['checks'].values())
    admission = json.loads(read(attempt['native_admission']))
    assert admission['candidate'] == candidate and admission['parent'] == parent
    assert admission['ranks'] == [35,100] and admission['mean_bill_reduction_pct_min'] == 0.1
    ng = json.loads(read(attempt['native_guard']))
    native_guard_check(ng, candidate)
    assert native_receipt['guard'] == ng
    guards.append(ng)
    reference, native = attempt['parent_native'], attempt['current_native']
    pm, nm = (json.loads(read(z['manifest.json'])) for z in (reference,native))
    pr, nr = lines(reference['rows.jsonl']), lines(native['rows.jsonl'])
    for manifest, source, rows in ((pm,parent,pr),(nm,candidate,nr)):
        assert manifest['candidate'] == source['path'] and manifest['sha256'] == source['sha256']
        assert manifest['ranks'] == [35,100] and [r['rank'] for r in rows] == [35,100]
    setup(nm['setup']['setup_s'], nm['setup']['pb_log'])
    native_parity = []
    for old,new,rank in zip(pr,nr,(35,100)):
        assert prediction(reference[f'pred_{rank}.npy']).tobytes() == prediction(native[f'pred_{rank}.npy']).tobytes()
        assert 0 < new['flops'] <= old['flops']
        native_parity.append(dict(rank=rank, saved_flops=old['flops']-new['flops'], all16_bytes_exact=True))
    bill_reduction = 1 - sum(r['flops'] for r in nr)/sum(r['flops'] for r in pr)
    assert bill_reduction >= 0.001
    assert native_parity[0]['saved_flops'] == build['pricing']['predicted_rank35_first_bind_reduction']

    worker = H2/(TAG+'_worker')
    wm, wl, wr = load(worker/'manifest.json'), load(worker/'launcher.json'), lines(pin(worker/'rows.jsonl'))
    assert wm['candidate'] == candidate['path'] and wm['sha256'] == candidate['sha256']
    assert wm['ranks'] == [35,100] and [r['rank'] for r in wr] == [35,100] and wm['server_instrumented']
    setup(wm['setup_s'], wm['pb_log'])
    assert wl['candidate_sha256'] == candidate['sha256'] and wl['client_exit'] == 0 and wl['server_retired']
    handles = wl['handles']
    assert handles['array_count_limit'] == 70000 and handles['peak_handles'] < 70000
    assert handles['stores'] and all(r['clears'] == 0 for r in handles['stores'])
    for i,rank in enumerate((35,100)):
        assert wr[i]['finite_outputs'] == 16384
        assert len(wr[i]['index_cache_entries']) == 3
        assert all(0 <= n <= cap for n,cap in zip(wr[i]['index_cache_entries'], (320,352,320)))
        meta = load(worker/f'pred_{rank}.json')
        assert meta['shape'] == [16,1024] and np.dtype(meta['dtype']) == np.dtype('float32')
        rpc = np.frombuffer(read(pin(worker/f'pred_{rank}.bin')), dtype=np.dtype(meta['dtype'])).reshape(16,1024)
        assert np.isfinite(rpc).all()
        assert rpc.astype(np.float64).tobytes() == prediction(native[f'pred_{rank}.npy']).tobytes()
        assert wr[i]['flops'] == nr[i]['flops']

    resolution = json.loads(read(attempt['baseline_resolution']))
    assert resolution['passed'] and resolution['candidate'] == parent
    assert resolution['fresh_candidate'] == attempt['baseline']
    base = attempt['baseline']
    bg, bl = json.loads(read(base['guard'])), json.loads(read(base['launcher']))
    assert bl['candidate'] == parent['path'] and bl['candidate_sha256'] == parent['sha256']
    assert bl['client_exit'] == 0 and bl['server_retired']
    base_tag = Path(base['census']['path']).name.removesuffix('_census.json')
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
    for i,(actual,old) in enumerate(zip(current,baseline)):
        assert actual['flops'] == nr[i]['flops'] and old['flops'] == pr[i]['flops']
        comparisons.append(dict(rank=actual['rank'], request_delta=actual['requests']-old['requests'],
            residual_s=actual['residual_s'], parent_residual_s=old['residual_s'],
            residual_delta_s=actual['residual_s']-old['residual_s'],
            server_cpu_ratio=actual['server_cpu_s']/old['server_cpu_s']))
    cpu_ratio = sum(r['server_cpu_ratio'] for r in comparisons)/2
    residual_proxy = 0.285450751 + 4*max(0.0, *(r['residual_delta_s'] for r in comparisons))
    assert cpu_ratio <= 1.04 and residual_proxy <= 0.35
    stress = load(H2/(TAG+'_setup')/'summary.json')
    assert stress['candidate_sha256'] == candidate['sha256']
    assert [r['repeat'] for r in stress['records']] == [0,1,2]
    for row in stress['records']:
        assert row['candidate'] == candidate['path'] and row['sha256'] == candidate['sha256'] and row['load_processes'] == 2
        setup(row['setup_s'], row['pb_log'])
    for record in list(EVIDENCE.values()):
        read(record)
    return dict(passed=True, candidate=candidate, parent=parent, native_parity=native_parity,
        mean_bill_reduction_pct=100*bill_reduction, worker_all16_promoted_bytes_exact=True,
        worker_native_flops_exact=True, boundary=comparisons, mean_cpu_ratio=cpu_ratio,
        official_residual_s=0.285450751, residual_proxy_s=residual_proxy,
        setup=dict(bound=651, fused=627, native_s=nm['setup']['setup_s'], worker_s=wm['setup_s'],
                   boundary_s=setup_rows[0]['setup_wall_s'], stress_s=[r['setup_s'] for r in stress['records']]),
        max_probe_rss_bytes=max(g['max_rss_bytes'] for g in guards), zero_added_swap=True,
        peak_handles=handles['peak_handles'], worker_networks=2, worker20_qualified=False,
        score_claim=False, submitted=False, evidence=list(EVIDENCE.values()), assessor=pin(__file__))


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
