"""Inert independent14 successor: --acquire --role bank|terminal, then --score.

All modes require --terminal-score-sha256 equal to the one advancing v2
terminal40 result. Both roles are freshly measured in separate OFFPY nice10
BLAS1 processes using the frozen native_v2 official-input/closed-bill/reset
loop. Only canonical PUBLIC8 + VIRGIN6 + repeat35 are acquired. The archive
is an exact bank OUTPUT/RAW control, never a substitute for fresh bills.

Fixed score: mean per-net final observed MSE and UNCLIPPED noise-corrected
MSE, each multiplied by max(.1, own complete F/2**41). Initialize once and
exclude repeat35. Retain only strict NC-adjusted gain on BOTH groups; no
per-net cost gate, refit, 24/40 transplant, milestone or release claim.
All outputs, complete bills, input/lifecycle receipts and sources are sealed
before independent4M truth hashes or arrays are read. Public35/100/200 were
TRAIN and150 VALID: independent MC is not unseen model generalization.
No private cache/budget mutation and no estimator-state reset is performed.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'panel_v2'
OLD = HERE / 'panel.py'
OLD_SHA = 'd364754060953a58f5b0d8b7d1ff2add18ae23f4b4384abf4882369ea01021a3'
NATIVE = HERE / 'native_v2.py'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
QUALIFIER = HERE / 'qualify_native_v2.py'
QUALIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
JUDGE_SHA = '82b098312660b97f813531989f0e94f0d313ada82a4f987e5a32bdbb8a724c73'
ADVANCE_SHA = '884089500b2f7f31cc9f1fbe343c6396df3cf976daee63141915715f8064c29f'
SCHEMA = 'birth_direction_terminal_independent14_v2'


def load(path, name, expected):
    import importlib.util
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen utility changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    old = load(OLD, 'bd_panel_v1_frozen_utilities', OLD_SHA)
    v2 = load(QUALIFIER, 'bd_panel_v2_lifecycle_verifier', QUALIFIER_SHA)
    h, p = old.helpers()
    # Inert namespace configuration only; all frozen files stay unchanged.
    old.JUDGE = HERE / 'terminal_score_v2'
    old.FIXED.update({OLD: OLD_SHA, NATIVE: NATIVE_SHA, QUALIFIER: QUALIFIER_SHA,
                     HERE / 'judge_terminal_v2.py': JUDGE_SHA,
                     Path(__file__).resolve(): h.digest(__file__)})
    return h, p, old, v2


def prerequisites(h, p, old, v2, terminal_sha):
    h.require(terminal_sha == ADVANCE_SHA, 'Only the frozen advancing terminal40 result')
    context = old.prerequisites(h, p, terminal_sha)
    pins, fixtures, rows, archive, old_score, candidate_sha = context
    judge = h.load_json(old.JUDGE / 'result.json')
    h.require(judge['schema'] == 'birth_direction_terminal_native40_v2' and
              judge['official_input_complete_bills_only'] and not judge['v1_partial_scores_used'] and
              judge['source_bindings'][str(HERE / 'judge_terminal_v2.py')] == JUDGE_SHA and
              judge['source_bindings'][str(NATIVE)] == NATIVE_SHA and
              candidate_sha == v2.TERMINAL_SHA, 'Repaired terminal40 and exact terminal checkpoint')

    def pin(path, expected=None):
        path = v2.path(path)
        h.require('/truthpost/' not in str(path) and 'mps_truth_' not in path.name and
                  not path.name.endswith('_truth4M.npz'), 'No truth in lifecycle qualification')
        value = h.binding(path)
        h.require(expected is None or value['sha256'] == expected, 'Lifecycle source/artifact pin')
        pins[str(path)] = value
        return value['sha256']

    v2.verify_development(pin, judge['development_qualification_sha256'])
    h.check_bindings(pins)
    return context


def replace_exact(original, replacements, require):
    changed = original
    for before, after in replacements:
        require(changed.count(before) == 1, 'One exact harness splice: ' + before[:90])
        changed = changed.replace(before, after, 1)
    inverse = changed
    for before, after in reversed(replacements):
        require(inverse.count(after) == 1, 'One exact inverse harness splice')
        inverse = inverse.replace(after, before, 1)
    require(inverse == original, 'Every undeclared native source byte is unchanged')
    return changed


def native_source(h):
    """Only fixture/receipt/output-control adaptation; the numeric loop is retained."""
    h.require(h.digest(NATIVE) == NATIVE_SHA, 'Frozen native-v2 loop')
    original = NATIVE.read_text()
    start = "    ranks=(35,100) if args.split=='development' else VALID\n"
    stop = "    args.out.mkdir()\n"
    h.require(original.count(start) == original.count(stop) == 1, 'Native fixture block boundaries')
    fixture_block = original[original.index(start):original.index(stop)]
    replacements = [
        (fixture_block,
         "    ranks=tuple(PANEL_RANKS)\n"
         "    rows={r['rank']:dict(weights=r['weights_path'],mlp_seed=r['mlp_seed']) for r in PANEL_ROWS}\n"
         "    inputs.update(PANEL_PINS)\n"
         "    inputs.update({v['path']:v['sha256'] for v in PANEL_FIXTURES.values()})\n"),
        ("    write(args.out/'started.json',result)\n",
         "    result.update(PANEL_HEADER)\n"
         "    write(args.out/'started.json',result)\n"
         "    result['started_sha256']=sha(args.out/'started.json')\n"),
        ("            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):\n",
         "            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):\n"
         "                signal.alarm(125)\n"),
        ("                # Save this attempted call before any resource/timing/repeat gate.\n",
         "                trial.update(panel_output_checks(np,values,call))\n"
         "                # Save this attempted call before any resource/timing/repeat gate.\n"),
        ("                guard(trial)\n",
         "                guard(trial)\n"
         "                assert trial['layer0_byte_exact'] and trial['archive_output_control_passed']\n"),
        ("                del mlp,served,materialized,values\n"
         "                gc.collect()\n",
         "                del mlp,served,materialized,values\n"
         "                gc.collect()\n"
         "                signal.alarm(0)\n"),
        ("        result.update(passed=True,hashes_unchanged=True,\n"
         "                      initialization_flops=result['trials'][0]['flops']-result['trials'][-1]['flops'])\n",
         "        panel_finish(result,args.out)\n"),
        ("    write(args.out/'completion.json',dict(passed=result['passed'],result_sha256=sha(args.out/'result.json')))\n",
         "    write(args.out/'completion.json',dict(passed=result['passed'],completed=result.get('completed',False),\n"
         "          result_sha256=sha(args.out/'result.json')))\n"),
    ]
    return replace_exact(original, replacements, h.require)


def acquire(h, p, old, v2, args):
    pins, fixtures, rows, archive, _, candidate_sha = prerequisites(h, p, old, v2, args.terminal_score_sha256)
    role = args.role
    candidate = old.BANK if role == 'bank' else old.CANDIDATE
    source = native_source(h)
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    ROOT.mkdir(exist_ok=True)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), 'Fixed nonsymlink panel root')
    directory = ROOT / role
    h.require(not directory.exists(), 'Exclusive role acquisition; never overwrite')

    def output_checks(np, values, call):
        prior = archive['trials'][call]
        reference = old.output(h, np, old.ARCHIVE / prior['output_artifact'], prior['output_bytes_sha256'])
        same = values.tobytes() == reference.tobytes()
        return dict(layer0_byte_exact=values[0].tobytes() == reference[0].tobytes(),
                    archive_output_byte_exact=same,
                    archive_output_control_passed=same if role == 'bank' else True,
                    served_materialized_byte_exact=True,
                    archived_output_bytes_sha256=prior['output_bytes_sha256'])

    def finish(result, folder):
        expected_init = 1383913666 + (120 if role == 'terminal' else 0)
        actual_init = result['trials'][0]['flops'] - result['trials'][-1]['flops']
        h.require(len(result['trials']) == 15 and actual_init == expected_init and
                  h.peak_rss_bytes() < 8_000_000_000, 'Complete14/repeat, exact init, final resource guard')
        for items in (pins, fixtures, archive['artifacts']):
            h.check_bindings(items)
        seal = dict(utc=h.utc(), source_bindings=result['source_bindings'], fixture_bindings=fixtures,
                    candidate_sha256=result['candidate_sha256'], role=role, trials=result['trials'],
                    input_preparation=result['input_preparation'], artifacts=result['artifacts'],
                    initialization_flops=actual_init, initialization_once=True, repeat_excluded=True,
                    truth_files_accessed=False)
        sealed_sha = h.write_json(folder / 'predictions_and_bills_sealed.json', seal)
        result.update(passed=True, completed=True, hashes_unchanged=True,
                      initialization_flops=actual_init, outputs_seal_sha256=sealed_sha)

    namespace = dict(__name__='bd_panel_v2_native_adapter', __file__=str(Path(__file__).resolve()),
                     PANEL_RANKS=p.RANKS, PANEL_ROWS=rows, PANEL_FIXTURES=fixtures,
                     PANEL_PINS={item['path']:item['sha256'] for item in pins.values()},
                     PANEL_HEADER=dict(schema=SCHEMA, role=role, completed=False, fixture_bindings=fixtures,
                        terminal_score_sha256=args.terminal_score_sha256, versions=old.VERSIONS,
                        pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS,0),
                        blas_env=dict.fromkeys(old.BLAS,'1'), executable=sys.executable,
                        no_warmup=True, instrumented=False, native_original_sha256=NATIVE_SHA,
                        native_adapter_sha256=source_sha, native_inverse_except_declared_harness_edits=True,
                        truth_files_accessed=False, archive_bills_used_for_scoring=False),
                     signal=signal, panel_output_checks=output_checks, panel_finish=finish)
    exec(compile(source, str(NATIVE), 'exec'), namespace)
    namespace['PINS'][NATIVE] = NATIVE_SHA
    saved = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), '--source', str(candidate),
                    '--split', 'validation', '--out', str(directory)]
        return namespace['main']()
    finally:
        signal.alarm(0)
        sys.argv = saved


def verifier(h, p, v2, fixtures, rows):
    """Reuse all native-v2 five-artifact/lifecycle checks with canonical panel rows."""
    original = QUALIFIER.read_text()
    node = next(x for x in ast.parse(original).body if isinstance(x, ast.FunctionDef) and x.name == 'verify_native')
    function = ''.join(original.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    start = "    rows = read(STORE / 'manifest.json')['roles']['observer_fit']['row_manifests']\n"
    stop = "    trials, preps = acq['trials'], acq['input_preparation']\n"
    h.require(function.count(start) == function.count(stop) == 1, 'Verifier fixture boundaries')
    block = function[function.index(start):function.index(stop)]
    changed = replace_exact(function, [(block,
        "    require(acq['fixture_bindings'] == PANEL_FIXTURES, 'Canonical independent14 fixtures')\n"
        "    weight_hashes = {}\n"
        "    for row in PANEL_ROWS:\n"
        "        rank = row['rank']\n"
        "        wp = path(row['weights_path'])\n"
        "        weight_hashes[rank] = PANEL_FIXTURES['weights_'+str(rank)]['sha256']\n"
        "        require(bindings[str(wp)] == pin(wp,weight_hashes[rank]), 'Canonical panel weights')\n"
        "    for item in PANEL_FIXTURES.values():\n"
        "        pin(path(item['path']),item['sha256'])\n")], h.require)
    namespace = dict(vars(v2), PANEL_FIXTURES=fixtures, PANEL_ROWS=rows)
    exec(compile(changed, str(QUALIFIER), 'exec'), namespace)
    return namespace['verify_native']


def verify_acquisition(h, p, old, v2, terminal_sha, role, context=None):
    context = context or prerequisites(h, p, old, v2, terminal_sha)
    pins, fixtures, rows, archive, _, candidate_sha = context
    directory = ROOT / role
    expected_candidate = old.BANK_SHA if role == 'bank' else candidate_sha
    r, end = h.load_json(directory/'result.json'), h.load_json(directory/'completion.json')
    h.require(end['passed'] and end['completed'] and end['result_sha256'] == h.digest(directory/'result.json') and
              r['passed'] and r['completed'] and r['schema'] == SCHEMA and r['role'] == role and
              r['candidate_sha256'] == expected_candidate and r['terminal_score_sha256'] == terminal_sha and
              r['versions'] == old.VERSIONS and not r['truth_files_accessed'] and
              not r['archive_bills_used_for_scoring'] and r['native_original_sha256'] == NATIVE_SHA and
              r['native_adapter_sha256'] == hashlib.sha256(native_source(h).encode()).hexdigest() and
              r['native_inverse_except_declared_harness_edits'], 'Sealed official-input panel role')
    actual = r['source_bindings']
    h.require(all(actual.get(item['path']) == item['sha256'] for item in pins.values()), 'All prospective panel pins')
    sealed = {}

    def pin(path, expected=None):
        path = v2.path(path)
        h.require('/truthpost/' not in str(path) and 'mps_truth_' not in path.name and
                  not path.name.endswith('_truth4M.npz'), 'No target before both acquisitions seal')
        item = h.binding(path)
        h.require(expected is None or item['sha256'] == expected, 'Fresh acquisition hash')
        sealed[str(path)] = item
        return item['sha256']

    outputs = verifier(h, p, v2, fixtures, rows)(r, directory, p.RANKS, pin)
    for call, trial in enumerate(r['trials']):
        prior = archive['trials'][call]
        old_output = old.ARCHIVE / prior['output_artifact']
        old_bytes = v2.payload(old_output, prior['output_bytes_sha256'], pin)
        h.require(trial['layer0_byte_exact'] and trial['served_materialized_byte_exact'] and
                  trial['archive_output_control_passed'] and outputs[call][:4096] == old_bytes[:4096] and
                  trial['archive_output_byte_exact'] == (outputs[call] == old_bytes) and
                  (role != 'bank' or outputs[call] == old_bytes), 'Actual bank bytes / terminal layer0')
    pin(directory/'started.json',r['started_sha256'])
    started = h.load_json(directory/'started.json')
    h.require(started['pid'] == r['pid'] and started['nice'] >= 10 and started['no_warmup'] and
              not started['instrumented'] and started['source_bindings'] == actual and
              started['fixture_bindings'] == fixtures and started['candidate_sha256'] == expected_candidate and
              started['blas_env'] == dict.fromkeys(old.BLAS,'1'), 'Start and complete-input ownership')
    for name in ('result.json','completion.json','predictions_and_bills_sealed.json'):
        pin(directory/name)
    h.require(h.digest(directory/'predictions_and_bills_sealed.json') == r['outputs_seal_sha256'], 'Output/bill seal')
    seal = h.load_json(directory/'predictions_and_bills_sealed.json')
    h.require(all(seal[k] == r[k] for k in ('source_bindings','fixture_bindings','candidate_sha256','role',
                                          'trials','input_preparation','artifacts','initialization_flops')) and
              not seal['truth_files_accessed'] and seal['initialization_once'] and seal['repeat_excluded'],
              'All fresh outputs and complete bills sealed before truth')
    h.check_bindings(sealed)
    return r, sealed, context


def score(h, p, old, v2, args, directory, result):
    context = prerequisites(h, p, old, v2, args.terminal_score_sha256)
    pins, fixtures, _, archive, archived_score, candidate_sha = context
    records, seals = {}, {}
    for role in ('bank','terminal'):
        records[role], role_seals, _ = verify_acquisition(h,p,old,v2,args.terminal_score_sha256,role,context)
        seals.update(role_seals)
    h.require(records['bank']['pid'] != records['terminal']['pid'], 'Separate fresh role processes')
    import numpy as np  # No estimator or FNP import in score mode.
    outputs = {role:[old.output(h,np,Path(t['output']),t['output_bytes_sha256'])
                     for t in r['trials']] for role,r in records.items()}
    for items in (pins,fixtures,seals):
        h.check_bindings(items)
    before = h.write_json(directory/'both_roles_verified_before_truth.json',dict(
        utc=h.utc(),source_bindings=pins,fixture_bindings=fixtures,role_seals=seals,
        candidate_sha256=candidate_sha,parent_sha256=old.BANK_SHA,ranks=list(p.RANKS),
        terminal_score_sha256=args.terminal_score_sha256,truth_files_accessed=False,
        initialization_once=True,repeat_excluded=True,archive_bills_used_for_scoring=False))
    result.update(source_bindings=pins,fixture_bindings=fixtures,role_seals=seals,
                  candidate_sha256=candidate_sha,parent_sha256=old.BANK_SHA,
                  both_roles_verified_before_truth_sha256=before,truth_bindings={},
                  per_role={'parent':[],'candidate':[]})
    index = h.load_json(old.TRUTH_INDEX)
    h.require(index['qualification_passed'] and index['hashes_unchanged'] and
              set(index['truth_bindings']) == {str(x) for x in p.RANKS}, 'Original independent4M truth index')
    result['truth_files_accessed'] = True
    h.write_json(directory/'truth_access_started.json',dict(utc=h.utc(),prior_seal_sha256=before))
    for rank in p.RANKS:
        path = p.G19 / (f'mps_truth_rank{rank:04d}_indep4M.npz' if rank in p.PUBLIC else f'fresh_{rank}_truth4M.npz')
        expected = index['truth_bindings'][str(rank)]
        h.require(Path(expected['path']).resolve() == path and h.digest(path) == expected['sha256'], 'Fixed4M truth hash')
        result['truth_bindings'][str(rank)] = h.binding(path)
    for call,rank in enumerate(p.RANKS):
        with np.load(result['truth_bindings'][str(rank)]['path'],allow_pickle=False) as src:
            mean,se = src['mc_post'][15].astype(np.float64),src['mc_se'][15].astype(np.float64)
        h.require(mean.shape == se.shape == (1024,) and np.isfinite(mean).all() and
                  np.isfinite(se).all() and (se >= 0).all(), 'Finite independent final truth')
        noise = float(np.mean(se*se))
        for label,role in (('parent','bank'),('candidate','terminal')):
            t = records[role]['trials'][call]
            raw = float(np.mean((outputs[role][call][15].astype(np.float64)-mean)**2))
            nc = raw-noise
            factor = max(.1,t['flops']/h.BUDGET)
            h.require(math.isfinite(raw) and raw >= 0 and math.isfinite(nc), 'Finite unclipped score')
            row = dict(rank=rank,flops=t['flops'],output_bytes_sha256=t['output_bytes_sha256'],
                       raw_observed=raw,raw_noise_corrected=nc,mc_se_squared_mean=noise,multiplier=factor,
                       adjusted_observed=raw*factor,adjusted_noise_corrected=nc*factor)
            if role == 'bank':
                old_row = archived_score['per_net'][call]
                # The old byte-inherited score contains these raw fields,
                # not a separately stored mc_se_squared_mean field.
                h.require(all(row[k] == old_row[k] for k in ('rank','raw_observed','raw_noise_corrected')),
                          'Exact archived bank RAW reproduction')
                row['old_bill_comparison_only'] = archive['trials'][call]['flops']
                row['actual_bill_difference_to_archive'] = t['flops']-archive['trials'][call]['flops']
            result['per_role'][label].append(row)
        print(json.dumps({role:items[-1] for role,items in result['per_role'].items()},allow_nan=False),flush=True)
    summaries = {}
    for role,items in result['per_role'].items():
        summaries[role] = {}
        for group,ranks in (('public8',p.PUBLIC),('virgin6',p.VIRGIN),('combined14',p.RANKS)):
            subset = [x for x in items if x['rank'] in ranks]
            fields = ('raw_observed','raw_noise_corrected','mc_se_squared_mean','flops',
                      'adjusted_observed','adjusted_noise_corrected')
            summaries[role][group] = dict(n=len(subset),**{
                'mean_'+key:sum(x[key] for x in subset)/len(subset) for key in fields})
    ratios,wins = {},{}
    for group in ('public8','virgin6','combined14'):
        b = summaries['parent'][group]['mean_adjusted_noise_corrected']
        c = summaries['candidate'][group]['mean_adjusted_noise_corrected']
        h.require(b > 0, 'Positive group comparator')
        ratios[group],wins[group] = c/b,c < b
    for items in (pins,fixtures,seals,result['truth_bindings']):
        h.check_bindings(items)
    h.require(h.digest(directory/'both_roles_verified_before_truth.json') == before and
              h.peak_rss_bytes() < 8_000_000_000, 'Score final seals/resources')
    result.update(passed=True,completed=True,qualification_passed=True,hashes_unchanged=True,
                  baseline_raw_recomputed_exact=True,baseline_output_bytes_exact=True,
                  archive_bills_used_for_scoring=False,fresh_official_input_complete_bills_only=True,
                  summaries=summaries,adjusted_candidate_over_parent=ratios,group_strict_gain_gate=wins,
                  retain_local_component=wins['public8'] and wins['virgin6'],
                  initialization_flops={k:r['initialization_flops'] for k,r in records.items()},repeat_excluded=True,
                  acquisition_sha256={k:h.digest(ROOT/k/'result.json') for k in records},
                  parent_score_sha256=old.BASE_SCORE_SHA,
                  aggregation='Mean per-net (final observed MSE - mean(mc_se15**2)) * max(.1, own fresh complete F/2**41); '
                  'unclipped; initialization once per role; repeated call14 excluded',
                  frame='Independent4M public8/virgin6 reused. Public35/100/200 TRAIN and150 VALID; not40 unseen holdouts.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--acquire',action='store_true')
    mode.add_argument('--score',action='store_true')
    parser.add_argument('--role',choices=('bank','terminal'))
    parser.add_argument('--terminal-score-sha256')
    args = parser.parse_args()
    if not args.acquire and not args.score:
        parser.print_help(); return 0
    if args.terminal_score_sha256 != ADVANCE_SHA or (args.acquire and not args.role) or (args.score and args.role):
        parser.error('Use the fixed advancing terminal-score SHA and --role only with --acquire')
    h,p,old,v2 = modules()
    old.runtime(h)

    def expired(_signal,_frame):
        raise TimeoutError('Fixed independent14 hard watchdog')

    signal.signal(signal.SIGALRM,expired)
    if args.acquire:
        return acquire(h,p,old,v2,args)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), 'Both fresh acquisitions are prerequisites')
    directory = ROOT/'score'
    directory.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA,mode='score',passed=False,completed=False,started_utc=h.utc(),
                  terminal_score_sha256=args.terminal_score_sha256,truth_files_accessed=False,
                  production_residual_validated=False,official_score=False,milestone_achieved=False,
                  goal_achieved=False,no_refit=True)
    began = time.monotonic()
    signal.alarm(300)
    try:
        score(h,p,old,v2,args,directory,result)
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
        result.update(completed_utc=h.utc(),wall_seconds=time.monotonic()-began,peak_rss_bytes=h.peak_rss_bytes())
        digest = h.write_json(directory/'result.json',result)
        h.write_json(directory/'completion.json',dict(passed=result['passed'],completed=result['completed'],
                     result_sha256=digest,retain_local_component=result.get('retain_local_component',False)))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','retain_local_component',
                                             'adjusted_candidate_over_parent','error')},allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
