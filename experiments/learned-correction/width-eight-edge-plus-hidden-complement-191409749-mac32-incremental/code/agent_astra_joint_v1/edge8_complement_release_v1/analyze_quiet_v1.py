"""Validate a completed quiet pair; never launches native work or release gates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
stamp = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    args = ap.parse_args()
    plan = read(args.manifest)
    for p, h in plan['source_pins'].items():
        assert sha(p) == h, ('changed pin', p)
    helper_path = next(Path(p) for p in plan['source_pins'] if Path(p).name == 'prepare_runtime_pair_parity_v3.py')
    spec = importlib.util.spec_from_file_location('quiet_validation', helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    out = Path(plan['quiet_output'])
    values = {}; evidence = {}
    for label, digest in [('bank', plan['sources']['bank_sha256']), ('candidate', plan['sources']['release_sha256'])]:
        dest = out / label
        guard = helper.valid_guard(dest / 'GUARD_v1.json', original=True)
        m, launcher = read(dest / 'manifest.json'), read(dest / 'launcher.json')
        assert m['sha256'] == launcher['sha256'] == digest
        assert m['ranks'] == [35, 150, 200] and m['client_flopscope'] == '0.12.3'
        assert m['diagnostic_harness_sha256'] == plan['source_pins'][str(helper.B / 'boundary_client_dual_v3.py')]
        assert m['thread_helper_sha256'] == plan['source_pins'][str(helper.B / 'thread_counts_v3.py')]
        assert launcher['client_exit'] == launcher['server_exit'] == 0 and launcher['handle_cap'] == 100000
        rr = helper.rows(dest / 'rows.jsonl')
        assert [r['rank'] for r in rr] == [35, 150, 200]
        for row, ref in zip(rr, plan['selected'][label]['references']):
            assert sha(ref['prediction']) == ref['prediction_sha256']
            assert helper.valid_row(row, dest / f"pred_{row['rank']}.bin") == Path(ref['prediction']).read_bytes()
            assert row['flops'] == ref['row']['flops'], ('own science tariff', label, row['rank'])
        values[label] = rr
        evidence[label] = dict(guard=guard, manifest_sha256=sha(dest / 'manifest.json'),
            guard_sha256=sha(dest / 'GUARD_v1.json'), rows_sha256=sha(dest / 'rows.jsonl'),
            setup=m['setup'], prediction_and_tariff_exact=True)
    compared = []
    for bank, cand in zip(values['bank'], values['candidate']):
        fields = ['wall_s', 'backend_s', 'thread_residual_instructions', 'requests']
        compared.append(dict(rank=bank['rank'], **{k + '_pct': 100 * (cand[k] / bank[k] - 1) for k in fields},
            bank={k: bank[k] for k in fields}, candidate={k: cand[k] for k in fields}))
    mean = {k: statistics.mean(x[k] for x in compared) for k in compared[0] if k.endswith('_pct')}
    report = dict(utc=stamp(), status='QUIET_THREE_RANK_OWN_SCIENCE_BYTES_AND_TARIFF_PASS',
        manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest), evidence=evidence,
        comparisons=compared, mean_relative_changes_pct=mean,
        scope='Exact source/science parity and original guards validated. Timing is descriptive, not a hosted safety guarantee or fire authorization.',
        release_min_run=False, submission_authorized=False)
    target = out / 'QUIET_PAIR_ANALYSIS_v1.json'
    with target.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=report['status'], mean=mean, output=str(target))))


if __name__ == '__main__':
    main()
