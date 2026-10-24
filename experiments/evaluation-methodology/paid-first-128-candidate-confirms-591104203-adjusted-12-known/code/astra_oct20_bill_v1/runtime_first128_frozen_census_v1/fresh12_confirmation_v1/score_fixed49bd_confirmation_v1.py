"""Whole12 confirmation of frozen49bd against measured337466; no fitting.

All candidate and reference cold groups must complete strict custody before
any truth is opened. The original four-source six contrasts remain immutable.
This reuses already observed truths, so it is confirmation, not unseen evidence.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
SOURCE = '49bd2ba31f0bce983f44cc001746e63ffaa84e530d3b31efa6424f69a0298c80'
BANK = '434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4'
GUARD = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
KEY = 'first128_paid_nocensus'


def strict_run(path, key, plan, plan_sha, validate_run):
    manifest = validate_run(path, key, plan, plan_sha)
    complete = read(path / 'COMPLETE.json')
    n = len(plan['groups'])
    assert not complete['pending'] and complete['plan_sha256'] == plan_sha
    assert len(complete['results']) == n and set(complete['results'].values()) == {0}
    assert manifest['max_active'] == 36 and manifest['retirement_cap_bytes'] == 6500000000
    assert manifest['empirical_admission_reservation_bytes'] == 4800000000
    for i in range(1, n + 1):
        part = path / f'part{i:02d}'
        g = read(path / f'part{i:02d}_GUARD.json')
        m, launcher, server = [read(part / name) for name in ('manifest.json', 'launcher.json', 'server.json')]
        assert g['guard_sha256'] == GUARD and g['reason'] is None
        assert g['successful'] and g['whole_group_retired'] and g['child_exit'] == 0 and not g['cleanup_errors']
        assert g['cgroup_events_initial'] == g['cgroup_events_final'] and not any(g['cgroup_events_final'].values())
        assert g['max_owned_swap_bytes'] == 0 and g['cgroup'] is True
        assert g['cap_bytes'] == 6500000000 and g['reserve_bytes'] == 19000000000
        assert g['max_active'] == 36 and g['admission_reservation_bytes'] == 4800000000
        assert g['peak_group_rss_bytes'] < 6500000000 and g['min_available_bytes'] >= 19000000000
        assert launcher['client_exit'] == launcher['server_exit'] == 0 and launcher['error'] is None
        assert not m['setup']['failed'] and server['environment']['sdk_source_hashes_verified']
        assert m['environment']['packages']['flopscope-client'] == '0.12.3'
        assert server['environment']['packages']['flopscope-server'] == '0.12.3'
        assert server['observed_process_threads'] == 4
        for row in [json.loads(s) for s in (part / 'rows.jsonl').read_text().splitlines()]:
            assert row['fast_rescue_stats'] == {'calls': 0, 'elements': 0}
            assert row['_SM40_COMPILE_STATS']['missing'] == row['_SM40_COMPILE_STATS']['capacity_skipped'] == 0
            assert row['_ASTRA_SM40_STATS']['owner_misses'] == 0 and row['_ASTRA_SM40_STATS']['retired']
            assert row['n_out'] == 16384 and row['dtype'] == 'float32'
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'packet', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--packet-sha', required=True)
    a = parser.parse_args()
    assert not a.out.exists() and sha(a.packet) == a.packet_sha
    packet = read(a.packet)
    assert sha(__file__) == packet['scorer_sha256']
    assert packet['candidate_source']['effective_sha256'] == SOURCE
    assert packet['confirmation_only'] and packet['all12_native_required_before_any_score']
    for rel, pin in packet['source_pins'].items():
        assert sha(a.root / rel) == pin, rel
    cf, mark = a.root / packet['cf_path'], a.root / packet['mark_path']
    sys.path[:0] = [str(cf), str(mark)]
    from grade_confirmation_v1 import validate_run, summary_with_mc
    from repair_four_source_plan_schema_v2 import verify_schema_repair
    from build_confirmation_plan_v1 import CORE_SHA
    from paired_mc_uncertainty_v1 import paired_report
    import numpy as np
    spec = importlib.util.spec_from_file_location('frozen_native_grader', mark / 'grade_linux_markline_v1.py')
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    loaded = []
    for cohort in packet['cohorts']:
        planpath, parentpath = [a.root / cohort[k]['path'] for k in ('plan', 'parent_plan')]
        assert sha(planpath) == cohort['plan']['sha256'] and sha(parentpath) == cohort['parent_plan']['sha256']
        plan, parent = read(planpath), read(parentpath)
        proofpath = a.root / cohort['parent_proof']['path']
        assert sha(proofpath) == cohort['parent_proof']['sha256']
        assert verify_schema_repair(a.root, parentpath, read(proofpath))
        expected = copy.deepcopy(parent)
        expected['candidates'][KEY] = packet['candidate_source']
        assert plan == expected, 'Only the frozen49bd candidate record may be added'
        seeds = cohort['seeds']
        assert plan['groups'] == [[seed] for seed in seeds]
        assert plan['first7'] == plan['held25'] == [] and plan['fixture_mode'] == 'fresh_selected_cold'
        assert plan['cpu_truth_core_sha256'] == CORE_SHA and plan['N'] == 64094208 and plan['N_half'] == 32047104
        assert parent['candidates']['public337466']['effective_sha256'] == BANK
        for rel, pin in plan['worker_source_pins'].items():
            assert sha(a.root / rel) == pin
        clock = plan['setup_clock']
        assert sha(a.root / clock['path']) == clock['sha256']
        runs = [Path(cohort['candidate_run']), Path(cohort['reference_run'])]
        strict_run(runs[0], KEY, plan, sha(planpath), validate_run)
        strict_run(runs[1], 'public337466', parent, sha(parentpath), validate_run)
        candidate, cs, cm = grader.collect(runs[0], fresh=False)
        bank, bs, bm = grader.collect(runs[1], fresh=False)
        assert cs == SOURCE and bs == BANK and cm == bm == ('paced', 4)
        assert set(candidate) == set(bank) == set(seeds)
        loaded.append((cohort, plan, candidate, bank))
    assert [seed for c, _, _, _ in loaded for seed in c['seeds']] == list(range(7109001, 7109013))
    # All24 candidate/reference groups have passed before any truth values open.
    rows = []
    summaries = {}
    for cohort, plan, candidate, bank in loaded:
        cohort_rows = []
        for seed in cohort['seeds']:
            c, b = candidate[seed], bank[seed]
            assert c['group'] == b['group'] == [seed]
            ra, rb = [plan['rows'][str(seed)]['truth_' + half] for half in ('A', 'B')]
            assert ra['path'] == rb['path'] and ra['sha256'] == rb['sha256']
            with np.load(grader.verified(a.root, ra), allow_pickle=False) as z:
                A, B, means = z[ra['key']].astype(np.float64), z[rb['key']].astype(np.float64), z['batch_final_means']
                assert int(z['seed']) == seed and str(z['cpu_protocol_sha256']) == CORE_SHA
                assert int(z['N']) == 64094208 and int(z['N_half']) == 32047104
            u1 = np.mean((c['prediction'] - A) * (c['prediction'] - B), axis=1)
            u0 = np.mean((b['prediction'] - A) * (b['prediction'] - B), axis=1)
            assert u1[15] > 0 and u0[15] > 0
            fr = c['row']['flops'] / b['row']['flops']
            row = dict(rank=seed, raw_pct=float(100 * (u1[15] / u0[15] - 1)),
                bill_pct=100 * (fr - 1), adjusted_pct=float(100 * (u1[15] / u0[15] * fr - 1)),
                u1=float(u1[15]), u0=float(u0[15]), flops=c['row']['flops'], bank_flops=b['row']['flops'],
                prediction_sha256=c['prediction_sha256'], bank_prediction_sha256=b['prediction_sha256'], truth_sha256=ra['sha256'])
            mc = paired_report(c['prediction'][15], b['prediction'][15], A[15], B[15], means,
                               c['row']['flops'], b['row']['flops'])
            assert mc['candidate_u'] == row['u1'] and mc['reference_u'] == row['u0']
            np.testing.assert_allclose(mc['adjusted_pct'], row['adjusted_pct'], rtol=0, atol=1e-12)
            row.update(paired_mc_se_pct=mc['paired_mc_se_pct'], paired_mc_details=mc)
            cohort_rows.append(row)
        summaries[cohort['name']] = summary_with_mc(grader, cohort_rows)
        rows.extend(cohort_rows)
    assert [r['rank'] for r in rows] == list(range(7109001, 7109013))
    result = dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
        status='COMPLETE_FIXED49BD_WHOLE12_CONFIRMATION', candidate_sha256=SOURCE, reference_sha256=BANK,
        reference_submission=337466, packet_sha256=a.packet_sha, scorer_sha256=sha(__file__),
        all=summary_with_mc(grader, rows), separate_cohort_summaries=summaries, rows=rows,
        actual_native_invoices=True, all24_original_guards_verified_before_truth=True,
        zero_child_memory_events_and_swap=True, no_fit_or_subset_selection=True,
        original_four_source_reports_unchanged=True, confirmation_only=True, unseen_claim=False,
        source_frozen_before_original_scores=True, release_qualified=False,
        uncertainty='Across-network SE and paired first-order Monte Carlo SE are separate.')
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(a.out), sha256=sha(a.out), all=result['all'], separate_cohort_summaries=summaries)))


if __name__ == '__main__':
    main()
