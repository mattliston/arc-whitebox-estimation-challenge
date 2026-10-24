"""Strict paired CPU confirmation against actual777a, with separate MC uncertainty.

Reuses the frozen native collector and summary. Numerical score is the same
float64 two-half U ratio times actual paired FLOPs; no inferred bank offset.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
from build_confirmation_plan_v1 import sha, read, utc, CORE_SHA, GUARD_SHA
from paired_mc_uncertainty_v1 import paired_report

HERE = Path(__file__).resolve().parent
MARK = HERE.parent / 'cloud_linux_markline_v1'
GRADER_SHA = '602cfb761c7058816c7998cf19737e0c011f13f47934369ed9f30e8dbc4ea5f2'
CLIENT_SHA = '1beba822cd02cebdfe9f8b30d0e08fb498a870a85fecaeccbe2fa38782098455'
MC_SHA = '5ec5fe3db031954baf48dce5b8db87b8ad1f448112b380a62d1470e621a1d4bb'


def validate_run(directory, key, plan, plan_sha):
    top = read(directory / 'RUN_MANIFEST.json')
    complete = read(directory / 'COMPLETE.json')
    assert complete['successful']
    assert top['candidate_sha256'] == plan['candidates'][key]['effective_sha256']
    assert top['plan_sha256'] == plan_sha
    assert top['groups'] == list(range(1, len(plan['groups']) + 1))
    assert top['blas_threads'] == 4 and top['setup'] == 'paced' and top['warmup_each'] is False
    assert sorted(p.name for p in directory.glob('part[0-9][0-9]')) == [
        f'part{i:02d}' for i in range(1, len(plan['groups']) + 1)]
    for i, group in enumerate(plan['groups'], 1):
        folder = directory / f'part{i:02d}'
        m = read(folder / 'manifest.json')
        g = read(directory / f'part{i:02d}_GUARD.json')
        launcher = read(folder / 'launcher.json')
        assert not m.get('diagnostic_only', False)
        assert m['sha256'] == top['candidate_sha256'] and m['ranks'] == group
        assert m['plan_sha256'] == plan_sha and m['harness_sha256'] == CLIENT_SHA
        assert m['warmup_each'] is False and m['setup_mode'] == 'paced'
        threads = m['environment']['thread_provenance']
        assert threads['blas_threads'] == 4 and set(threads['environment'].values()) == {'4'}
        assert m['environment']['sdk_source_hashes_verified']
        assert g['guard_sha256'] == GUARD_SHA and g['cgroup'] is True
        assert g['successful'] and g['whole_group_retired'] and g['child_exit'] == 0
        assert not g['cleanup_errors'] and g.get('max_owned_swap_bytes', 0) == 0
        assert threads == g['thread_provenance'] == launcher['thread_provenance']
        assert launcher['client_exit'] == 0 and launcher['error'] is None
        records = [json.loads(line) for line in (folder / 'rows.jsonl').read_text().splitlines() if line.strip()]
        assert len(records) == 1 and records[0]['rank'] == group[0]
        row = records[0]
        assert row['phase'] == 'scored' and row['nonfinite'] == 0
        assert row.get('fast_rescue_stats', {}).get('calls', 0) == 0
        pred = folder / f'pred_{group[0]}.bin'
        assert pred.stat().st_size == 65536 and sha(pred) == row['prediction_sha256']
        assert m['fixture_hashes'][str(group[0])] == plan['rows'][str(group[0])]['weights']['sha256']
    return top


def summary_with_mc(grader, rows):
    out = grader.summary(rows)
    out['paired_mc_se_of_mean_pct'] = math.sqrt(sum(r['paired_mc_se_pct'] ** 2 for r in rows)) / len(rows)
    out['mc_se_is_separate_from_across_network_se'] = True
    return out


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'plan', 'candidate', 'bank', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--candidate-key', default='edge8_nonlinear')
    args = p.parse_args()
    assert not args.out.exists()
    assert sha(MARK / 'grade_linux_markline_v1.py') == GRADER_SHA
    assert sha(HERE / 'paired_mc_uncertainty_v1.py') == MC_SHA
    sys.path.insert(0, str(MARK))
    spec = importlib.util.spec_from_file_location('frozen_native_grade', MARK / 'grade_linux_markline_v1.py')
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    plan = grader.load_plan(args.root, args.plan)
    proof = read(args.plan.with_suffix('.PROOF.json'))
    assert proof['successful'] and proof['plan_sha256'] == sha(args.plan)
    assert proof['builder_sha256'] == sha(HERE / 'build_confirmation_plan_v1.py')
    assert plan['fixture_mode'] == 'fresh_selected_cold' and plan['first7'] == plan['held25'] == []
    assert plan['reference_submission_id'] == 337422 and plan['reference_candidate'] == 'bank337422'
    assert plan['cpu_truth_core_sha256'] == CORE_SHA
    assert args.candidate_key == 'edge8_nonlinear'
    for rel, pin in plan['worker_source_pins'].items():
        assert sha(args.root / rel) == pin
    cm = validate_run(args.candidate, args.candidate_key, plan, sha(args.plan))
    bm = validate_run(args.bank, 'bank337422', plan, sha(args.plan))
    candidate, cs, c_mode = grader.collect(args.candidate, fresh=False)
    bank, bs, b_mode = grader.collect(args.bank, fresh=False)
    ranks = [r for group in plan['groups'] for r in group]
    assert set(candidate) == set(bank) == set(ranks) and c_mode == b_mode == ('paced', 4)
    assert cs == cm['candidate_sha256'] and bs == bm['candidate_sha256']
    result = []
    for rank in ranks:
        c, b = candidate[rank], bank[rank]
        assert c['group'] == b['group'] == [rank]
        a_record, b_record = [plan['rows'][str(rank)]['truth_' + half] for half in ('A', 'B')]
        assert a_record['path'] == b_record['path'] and a_record['sha256'] == b_record['sha256']
        truth = grader.verified(args.root, a_record)
        with np.load(truth, allow_pickle=False) as z:
            A, B = z[a_record['key']].astype(np.float64), z[b_record['key']].astype(np.float64)
            means = z['batch_final_means']
            assert int(z['seed']) == rank and str(z['cpu_protocol_sha256']) == CORE_SHA
        # Exact same score expressions as frozen grade_linux_markline_v1.
        u1 = np.mean((c['prediction'] - A) * (c['prediction'] - B), axis=1)
        u0 = np.mean((b['prediction'] - A) * (b['prediction'] - B), axis=1)
        assert u0[15] > 0 and u1[15] > 0
        fr = c['row']['flops'] / b['row']['flops']
        row = dict(rank=rank, raw_pct=float(100 * (u1[15] / u0[15] - 1)), bill_pct=100 * (fr - 1),
                   adjusted_pct=float(100 * (u1[15] / u0[15] * fr - 1)), u1=float(u1[15]), u0=float(u0[15]),
                   flops=c['row']['flops'], bank_flops=b['row']['flops'],
                   requests=c['row']['requests'], bank_requests=b['row']['requests'],
                   prediction_sha256=c['prediction_sha256'], bank_prediction_sha256=b['prediction_sha256'],
                   max_abs_dy=float(np.max(np.abs(c['prediction'] - b['prediction']))),
                   layer_raw_pct=[100 * (float(x) / float(y) - 1) if y else None for x, y in zip(u1, u0)])
        mc = paired_report(c['prediction'][15], b['prediction'][15], A[15], B[15], means,
                           c['row']['flops'], b['row']['flops'])
        assert mc['candidate_u'] == row['u1'] and mc['reference_u'] == row['u0']
        np.testing.assert_allclose(mc['adjusted_pct'], row['adjusted_pct'], rtol=0, atol=1e-12)
        row.update(paired_mc_se_pct=mc['paired_mc_se_pct'], paired_mc_details=mc)
        result.append(row)
    report = dict(utc=utc(), reference='paired same-host exact source for submission337422',
        candidate_sha256=cs, bank_sha256=bs, plan_sha256=sha(args.plan), setup_mode=c_mode,
        all=summary_with_mc(grader, result), rows=result,
        confirmation_splits={name: summary_with_mc(grader, [r for r in result if r['rank'] in seeds])
                             for name, seeds in plan['confirmation_splits'].items()},
        source_grader_sha256=GRADER_SHA, grader_sha256=sha(__file__), mc_helper_sha256=MC_SHA,
        independent_cpu_truth=True, no_fit_or_model_selection=True, release_qualified=False,
        uncertainty='Across-network standard errors and paired first-order MC error are reported separately; neither is a release guarantee.')
    grader.write_new(args.out, report)
    print(json.dumps(dict(all=report['all'], confirmation_splits=report['confirmation_splits'])))


if __name__ == '__main__':
    main()
