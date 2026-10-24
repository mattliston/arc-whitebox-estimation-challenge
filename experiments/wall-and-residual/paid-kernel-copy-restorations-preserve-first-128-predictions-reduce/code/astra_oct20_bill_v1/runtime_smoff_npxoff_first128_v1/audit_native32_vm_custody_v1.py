"""Independently verify the completed original first128+SMoff+NPoff VM32 custody."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
O = HERE.parents[3]
A = O / 'lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1'
L = A / 'cloud_linux_markline_v1'
C = HERE / 'native32_vm_custody_v1'
N = HERE
SOURCE = 'fe8cadf30daf0921a86acab95ec91348a565ce64f52e3dd4522a1936ac60620b'
PLAN = '9ec4c570e5f0846456af14af84eb225bf01d4c809567455bd334d235abda153d'
GUARD = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_text())

copy = read(C / 'COPY_RECEIPT_v1.json')
assert len(copy['files']) == 95
for item in copy['files']:
    p = C / item['path']
    assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
assert sha(C / 'candidate_first128_smoff_npxoff_science_v1.py') == SOURCE
assert sha(N / 'candidate_first128_smoff_npxoff_science_v1.py') == SOURCE
assert sha(N / 'NATIVE_PLAN_LINUX_v1.json') == PLAN
plan = read(N / 'NATIVE_PLAN_LINUX_v1.json')
complete, run = read(C / 'COMPLETE.json'), read(C / 'RUN_MANIFEST.json')
assert complete['successful'] and not complete['pending']
assert len(complete['results']) == 12 and set(complete['results'].values()) == {0}
assert complete['plan_sha256'] == run['plan_sha256'] == PLAN
assert complete['blas_threads'] == run['blas_threads'] == 4
assert complete['max_active'] == run['max_active'] == 36
assert run['candidate_sha256'] == SOURCE and run['jobs'] == 12
assert run['retirement_cap_bytes'] == 6500000000
assert run['empirical_admission_reservation_bytes'] == 4800000000
assert run['setup'] == 'paced' and run['warmup_each'] is False
report_path = L / 'local337466_auto_v1/reports/scores_edge8_complement_first128_local90_smoff_npxoff_v1__edge8_complement_first128_local90_smoff_npxoff_FULL32_VS337466_v1.json'
report = read(report_path)
assert report['candidate_sha256'] == SOURCE and report['reference_submission'] == 337466
assert report['reference_sha256'] == '434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4'
score_rows = {r['rank']: r for r in report['rows']}
rows, guards = [], []
for number, ranks in enumerate(plan['groups'], 1):
    part = C / f'part{number:02d}'
    g = read(C / f'part{number:02d}_GUARD.json')
    guards.append(g)
    assert g['guard_sha256'] == GUARD and g['successful'] and g['child_exit'] == 0
    assert g['reason'] is None and g['whole_group_retired'] and not g['cleanup_errors']
    assert g['max_owned_swap_bytes'] == 0 and g['cap_bytes'] == 6500000000
    assert g['reserve_bytes'] == 19000000000 and g['max_active'] == 36
    assert g['admission_reservation_bytes'] == 4800000000 and g['cgroup'] is True
    assert g['thread_provenance']['blas_threads'] == 4
    assert g['cgroup_events_final'] == g['cgroup_events_initial']
    assert all(v == 0 for v in g['cgroup_events_final'].values())
    assert g['peak_group_rss_bytes'] < 6500000000 and g['min_available_bytes'] >= 19000000000
    m, launch, server = read(part / 'manifest.json'), read(part / 'launcher.json'), read(part / 'server.json')
    assert m['sha256'] == launch['source_sha256'] == SOURCE
    assert m['plan_sha256'] == launch['plan_sha256'] == PLAN and m['ranks'] == ranks
    assert m['harness_sha256'] == '1beba822cd02cebdfe9f8b30d0e08fb498a870a85fecaeccbe2fa38782098455'
    assert not m['setup']['failed'] and m['setup_mode'] == launch['setup_mode'] == 'paced'
    assert m['warmup_each'] is launch['warmup_each'] is False
    assert m['setup_is_release_gate'] is launch['release_gate'] is False
    assert launch['client_exit'] == launch['server_exit'] == 0 and launch['error'] is None
    assert launch['handle_cap'] == 100000 and launch['array_bytes_cap'] == 4294967296
    assert server['environment']['sdk_source_hashes_verified']
    assert m['environment']['sdk_source_hashes_verified']
    assert m['environment']['packages']['flopscope-client'] == '0.12.3'
    assert server['environment']['packages']['flopscope-server'] == '0.12.3'
    assert server['observed_process_threads'] == 4
    part_rows = [json.loads(line) for line in (part / 'rows.jsonl').read_text().splitlines()]
    assert [r['rank'] for r in part_rows] == ranks
    for row in part_rows:
        assert row['phase'] == 'scored' and row['nonfinite'] == 0
        assert row['n_out'] == 16384 and row['dtype'] == 'float32'
        assert row['fast_rescue_stats'] == {'calls': 0, 'elements': 0}
        assert row['_SM40_COMPILE_STATS']['missing'] == row['_SM40_COMPILE_STATS']['capacity_skipped'] == 0
        assert row['_ASTRA_SM40_STATS']['owner_misses'] == 0 and row['_ASTRA_SM40_STATS']['retired']
        pred = part / f"pred_{row['rank']}.bin"
        sr = score_rows[row['rank']]
        assert pred.stat().st_size == 65536
        assert sha(pred) == row['prediction_sha256'] == sr['prediction_sha256']
        assert row['flops'] == sr['flops']
    rows.extend(part_rows)
assert len(rows) == len(score_rows) == 32
assert set(score_rows) == set(plan['first7'] + plan['held25'])
increments = read(L / 'first128_smoff_npxoff_increment_v1/FULL32_VS_FIRST1284DA_v1.json')
assert increments['candidate_report_sha256'] == sha(report_path)
for row in increments['rows']:
    sr = score_rows[row['rank']]
    assert row['candidate_prediction_sha256'] == sr['prediction_sha256']
    assert row['candidate_flops'] == sr['flops'] and row['candidate_u'] == sr['u1']
    calc = 100 * (row['candidate_u'] / row['parent_u'] * row['candidate_flops'] / row['parent_flops'] - 1)
    assert abs(calc - row['adjusted_pct']) < 1e-11
assert abs(statistics.mean(r['adjusted_pct'] for r in increments['rows']) - increments['all']['adjusted_pct']) < 1e-11
parent_custody = HERE.parent / 'breakthrough_v1/first_hidden_readout_v1/native32_vm_custody_v1'
parent_copy = read(parent_custody / 'COPY_RECEIPT_v1.json')
for item in parent_copy['files']:
    p = parent_custody / item['path']
    assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
assert sha(parent_custody / 'INDEPENDENT_CUSTODY_AUDIT_v1.json') == 'f5de32b792aa06abea939d225cbc8f59c2a2c6ca327fdc54ea2d4a415d245c29'
parent_rows, avoided = {}, {}
for part in sorted(parent_custody.glob('part[0-9][0-9]')):
    last = 0
    for line in (part / 'rows.jsonl').read_text().splitlines():
        r = json.loads(line)
        parent_rows[r['rank']] = r
        assert '_O20_NPX_COPY_EXACT_STATS' not in r
        now = r['_O20_SMDIRECT_STATS']['avoided_stack_elements']
        avoided[r['rank']] = now - last
        last = now
assert set(parent_rows) == set(score_rows)
tariff_rows = []
for r in rows:
    parent = parent_rows[r['rank']]
    assert r['prediction_sha256'] == parent['prediction_sha256']
    assert r['_O20_SMDIRECT_STATS'] == {key: 0 for key in parent['_O20_SMDIRECT_STATS']}
    debit = r['flops'] - parent['flops']
    assert debit >= avoided[r['rank']] and debit > 0, (r['rank'], debit, avoided[r['rank']])
    tariff_rows.append(dict(rank=r['rank'], added_flops=debit,
        parent_avoided_stack_elements=avoided[r['rank']],
        measured_total_minus_predicted_stack_debit=debit-avoided[r['rank']],
        prediction_sha256=r['prediction_sha256']))
assert increments['all']['raw_pct'] == increments['held25']['raw_pct'] == 0.0

scoring_path = HERE / 'SCORING_PLAN_LINUX_v1.json'
compatibility_path = HERE / 'SCORING_PLAN_COMPATIBILITY_v1.json'
compatibility = read(compatibility_path)
scoring_plan = read(scoring_path)
frozen_bank = scoring_plan['candidates'].pop('bank337393')
assert scoring_plan == plan
assert frozen_bank['effective_sha256'] == 'af2023a220f24299bdf91e464ce809a51159b002e178a8f47dfad3fba7b4eb0a'
official_report_path = Path(report['candidate_official_report'])
assert sha(official_report_path) == report['candidate_official_report_sha256']
assert read(official_report_path)['plan_sha256'] == sha(scoring_path) == compatibility['scoring_plan_sha256']
assert compatibility['native_plan_sha256'] == PLAN

result = dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
    status='PASS_ORIGINAL_VM32_CUSTODY', source_sha256=SOURCE, plan_sha256=PLAN,
    copy_receipt_sha256=sha(C/'COPY_RECEIPT_v1.json'), files_rehashed=95,
    rows_verified=32, guards_verified=12, original_guard_sha256=GUARD,
    source_identical_local_remote=True, all_prediction_hashes_match_scores=True,
    zero_rescue_missing_skipped_owner_miss=True, zero_swap_oom=True,
    max_group_rss_bytes=max(g['peak_group_rss_bytes'] for g in guards),
    max_cgroup_memory_bytes=max(g['peak_cgroup_memory_bytes'] for g in guards),
    min_available_bytes=min(g['min_available_bytes'] for g in guards),
    completed_utc=complete['utc'], current_best_report=str(report_path),
    current_best_report_sha256=sha(report_path), current_best_all=report['all'],
    current_best_held25=report['held25'], first1284da_increment_all=increments['all'],
    first1284da_increment_held25=increments['held25'], per_rank_ratio_comparison=True,
    original_caps_unchanged=True, runtime_qualified=False, release_qualified=False,
    parent_custody_sha256=sha(parent_custody / 'INDEPENDENT_CUSTODY_AUDIT_v1.json'),
    all_predictions_byte_identical_to_first1284da=True,
    actual_total_paid_debit_includes_stack_and_np_off=True,
    stack_component_predicted_from_proved_one_flop_tariff=True,
    np_counter_absent_from_original_native_rows=True,
    no_independent_np_counter_decomposition_claimed=True,
    actual_added_flops_min=min(r['added_flops'] for r in tariff_rows),
    actual_added_flops_max=max(r['added_flops'] for r in tariff_rows),
    actual_added_flops_sum=sum(r['added_flops'] for r in tariff_rows),
    tariff_rows=tariff_rows,
    scoring_plan_sha256=sha(scoring_path),
    scoring_compatibility_sha256=sha(compatibility_path),
    only_score_plan_bank_metadata_added=True,
    measured_total_minus_predicted_stack_debit_sum=sum(r['measured_total_minus_predicted_stack_debit'] for r in tariff_rows),
    audit_source_sha256=sha(Path(__file__)))
dest = C / 'INDEPENDENT_CUSTODY_AUDIT_v1.json'
with dest.open('x') as f:
    json.dump(result, f, indent=2); f.write('\n')
print(json.dumps(dict(path=str(dest), sha256=sha(dest), **result)))
