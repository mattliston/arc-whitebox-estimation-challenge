"""Inert fixed VALID40 autonomous-cap native judge; explicit completed seals only.

No estimator, Torch or FNP import. The unchanged current-bank official-input
acquisition is reused, not its older NumPy-input bills. Both capped null and
terminal use their OWN complete new bills. All source/payload/output/bill
checks and the pre-truth seal precede any VALID truth bytes. No fit/refit.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from common import BD, HERE, NEW, P2, PARENT, PARENT_SHA, STORE, completed, module, sha, unchanged, write

BUDGET = 2**41
BANK_RESULT_SHA = '6b48bebac1442dca55d093efd194dc2852a770be3420b6995d4aa96f3aa7083a'
NULL_SHA = 'd6dc6bb47d0c1f047ae34c1fb37ff624ccb957e20074447725abe7b7c21fcbf4'
QUALIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
TRAINER_SHA = '6780e3e8f9f7ed3e6fddd0c347f6024f756972cf29bf3987e3622b08529cd81c'
TRUTH_INDEX = NEW/'joint_prefix_final_judge_cached_v2/receipt.json'
TRUTH_INDEX_SHA = '4968610553b95011cabdc651dd65d4c9057054f9d2d2bf73711426fff695893c'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def replace_exact(source, edits):
    changed = source
    for before, after in edits:
        require(changed.count(before) == 1, 'Exact verifier substitution: '+before[:80])
        changed = changed.replace(before, after, 1)
    inverse = changed
    for before, after in reversed(edits):
        require(inverse.count(after) == 1, 'Exact verifier inverse')
        inverse = inverse.replace(after, before, 1)
    require(inverse == source, 'Every undeclared verifier byte remains unchanged')
    return changed


def native_verifier(v2, terminal_sha):
    """Only candidate identity/init adaptation of the five-artifact verifier."""
    path = BD/'qualify_native_v2.py'
    require(sha(path) == QUALIFIER_SHA, 'Preserved complete-bill verifier')
    original = path.read_text()
    node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_native')
    source = ''.join(original.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    source = replace_exact(source, [
        ("    expected_sha = BANK_SHA if source == BANK else {HERE / 'build_zero_v1/estimator.py': ZERO_SHA,\n"
         "                       HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA}.get(source)\n",
         "    expected_sha = BANK_SHA if source == BANK else {CAP / 'estimator.py': CAP_NULL_SHA,\n"
         "                       CAP / 'build_terminal_v1/estimator.py': CAP_TERMINAL_SHA}.get(source)\n"),
        ("    expected_init = 1383913666 + (0 if source == BANK else 120)\n",
         "    expected_init = 1383913666  # Same P shapes; no birth-direction parameters.\n"),
    ])
    env = dict(vars(v2), CAP=HERE, CAP_NULL_SHA=NULL_SHA, CAP_TERMINAL_SHA=terminal_sha)
    exec(compile(source, str(__file__)+':preserved_verifier', 'exec'), env)
    return env['verify_native']


def judge(args, out, result):
    pins, truth_pins = {}, {}
    truth_paths = {r:P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{r:04d}.npz' for r in VALID}
    forbidden = {str(p.resolve()) for p in truth_paths.values()}

    def pin(path, expected=None):
        path = Path(path).resolve()
        require(str(path) not in forbidden, 'VALID truth accessed before prediction/bill seal')
        value = pins.get(str(path))
        if value is None:
            value = sha(path); pins[str(path)] = value
        require(expected is None or value == expected, 'Source/artifact hash: '+str(path))
        return value

    def bind(items):
        for path, value in items.items():
            if isinstance(value, dict):
                path, value = value['path'], value['sha256']
            pin(path, value)

    pin(__file__, args.source_sha256)
    pin(HERE/'common.py')
    pin(HERE/'train.py', TRAINER_SHA)
    pin(HERE/'build_terminal.py', args.builder_sha256)
    pin(HERE/'native40.py', args.native_source_sha256)
    pin(BD/'qualify_native_v2.py', QUALIFIER_SHA)
    pin(TRUTH_INDEX, TRUTH_INDEX_SHA)
    builder = module(HERE/'build_terminal.py', 'cap_frozen_terminal_builder_utilities')
    training, training_pins = builder.training_admission(args.training_sha256)
    bind(training_pins)
    require(training['source_bindings'][str(HERE/'train.py')] == TRAINER_SHA and
            training['validation_ranks'] == list(VALID) and training['updates'] == 240 and
            training['unused_layer0_parameters_unchanged'], 'Exact terminal-only completed fit')
    build_path = HERE/'build_terminal_v1/build.json'
    pin(build_path, args.build_sha256)
    build = read(build_path)
    require(build['passed'] and build['training_result_sha256'] == args.training_sha256 and
            build['trainer_sha256'] == TRAINER_SHA and build['parent_sha256'] == NULL_SHA and
            build['terminal_payload'] == training['terminal_payload'], 'Native terminal checkpoint ownership')
    bind(build['source_bindings'])
    terminal_source = HERE/'build_terminal_v1/estimator.py'
    terminal_sha = pin(terminal_source, build['candidate_sha256'])
    pin(HERE/'estimator.py', NULL_SHA)
    builder.inverse_check((HERE/'estimator.py').read_bytes(), terminal_source.read_bytes())
    v2 = module(BD/'qualify_native_v2.py', 'cap_frozen_native_lifecycle_verifier')
    verify = native_verifier(v2, terminal_sha)
    acquisitions, native_bytes = {}, {}
    definitions = {
        'bank': (BD/'validation_bank_v2', BANK_RESULT_SHA, PARENT, PARENT_SHA),
        'null': (HERE/'native_validation_null_v1', args.null_sha256, HERE/'estimator.py', NULL_SHA),
        'terminal': (HERE/'native_validation_terminal_v1', args.terminal_sha256, terminal_source, terminal_sha),
    }
    for role, (folder, digest, source, source_sha) in definitions.items():
        pin(folder/'result.json', digest); pin(folder/'completion.json')
        acq, end = read(folder/'result.json'), read(folder/'completion.json')
        require(acq['passed'] and end['passed'] and end['result_sha256'] == digest and
                Path(acq['source']).resolve() == source and acq['candidate_sha256'] == source_sha,
                'Qualified actual acquisition: '+role)
        bind(acq['source_bindings'])
        if role != 'bank':
            require(acq['schema'] == 'current_paircap4096_e2e_native40_v1' and acq['role'] == role and
                    acq['own_complete_bills'] and acq['all_outputs_and_bills_sealed_before_scoring'] and
                    acq['source_bindings'][str(HERE/'native40.py')] == args.native_source_sha256,
                    'Capped own complete source/cost role')
        native_bytes[role] = verify(acq, folder, VALID, pin)
        acquisitions[role] = acq
    require(all(len(v) == 41 for v in native_bytes.values()), 'All40 plus byte-exact repeat')

    import numpy as np  # Offline score arrays only, never the estimator or weights.
    def output(path, expected=None):
        value = np.load(path, allow_pickle=False)
        require(type(value) is np.ndarray and value.shape == (16,1024) and value.dtype == np.float32 and
                np.isfinite(value).all() and (value >= 0).all(), 'Finite served FP32 output')
        require(expected is None or hashlib.sha256(value.tobytes()).hexdigest() == expected, 'Served-byte ownership')
        return value

    predictions = {role:[np.frombuffer(data,dtype='<f4').reshape(16,1024) for data in values]
                   for role,values in native_bytes.items()}
    mps, drift = [], []
    for call, rank in enumerate(VALID):
        item = training['validation_outputs'][str(rank)]
        pin(item['path'], item['sha256'])
        shadow = output(item['path'], item['raw_sha256'])
        mps.append(shadow)
        for role in ('null','terminal'):
            require(predictions[role][call][0].tobytes() == predictions['bank'][call][0].tobytes(),
                    'Exact unaltered Gaussian layer0')
        delta = shadow.astype(np.float64)-predictions['terminal'][call].astype(np.float64)
        maximum = float(np.max(np.abs(delta)))
        require(maximum <= 1e-5, 'Terminal native/MPS full-size fidelity')
        drift.append(dict(rank=rank,max_absolute=maximum,final_rms=float(np.sqrt(np.mean(delta[-1]**2)))))
    # Actual terminal payload arrays also reproduce the epoch-six audit arrays.
    terminal_payload = training['terminal_payload']['path']
    audit_payload = HERE/'training_v1/epoch_06_audit_only.npz'
    pin(audit_payload); pin(terminal_payload,training['terminal_payload']['sha256'])
    with np.load(audit_payload,allow_pickle=False) as audit, np.load(terminal_payload,allow_pickle=False) as terminal:
        require(set(audit.files) == set(terminal.files), 'Epoch-six full payload keys')
        require(all(audit[k].dtype == terminal[k].dtype and audit[k].shape == terminal[k].shape and
                    audit[k].tobytes() == terminal[k].tobytes() for k in audit.files), 'No checkpoint selection')
    unchanged(pins)
    seal = dict(utc=utc(),source_bindings=dict(pins),requested_seals=vars(args),truth_accessed=False,
        training_validation_outputs=training['validation_outputs'],terminal_payload=training['terminal_payload'],
        native_trials={k:v['trials'] for k,v in acquisitions.items()},mps_native_drift=drift,
        initialization_once=True,repeat_excluded=True,bank_official_input_receipt_sha256=BANK_RESULT_SHA)
    write(out/'predictions_and_bills_sealed.json',seal)
    before = sha(out/'predictions_and_bills_sealed.json')
    result.update(source_bindings=dict(pins),all_predictions_verified_before_truth=True,
        predictions_and_bills_seal_sha256=before,mps_native_drift=drift,truth_accessed=True)
    records = read(TRUTH_INDEX)['records']
    require(len(records) == 40 and {x['rank'] for x in records} == set(VALID), 'Existing VALID40 truth index')
    expected_truths = {x['rank']:x['target_sha256'] for x in records}
    for rank,path in truth_paths.items():
        require(sha(path) == expected_truths[rank], 'Fixed1e9 truth hash')
        truth_pins[str(path)] = expected_truths[rank]
    write(out/'truth_bindings.json',truth_pins)
    rows = []
    for call,rank in enumerate(VALID):
        with np.load(truth_paths[rank],allow_pickle=False) as src:
            target = src['mc_post'].astype(np.float64)
        require(target.shape == (16,1024) and np.isfinite(target).all(), 'Finite fixed target')
        row = dict(rank=rank,per_role={},adjusted_wins={},raw_wins={})
        for role,acq in acquisitions.items():
            layer_mse = np.mean((predictions[role][call].astype(np.float64)-target)**2,axis=1)
            raw,flops = float(layer_mse[-1]),acq['trials'][call]['flops']
            multiplier = max(.1,flops/BUDGET)
            require(math.isfinite(raw) and raw >= 0, 'Finite actual raw objective')
            row['per_role'][role] = dict(raw_mse=raw,adjusted=raw*multiplier,flops=flops,
                multiplier=multiplier,layer_mse=layer_mse.tolist())
        mps_raw = float(np.mean((mps[call][-1].astype(np.float64)-target[-1])**2))
        native_raw = row['per_role']['terminal']['raw_mse']
        require(native_raw > 0 and .99 <= mps_raw/native_raw <= 1.01, 'Terminal native/MPS MSE within1%')
        row.update(mps_raw_mse=mps_raw,mps_native_mse_ratio=mps_raw/native_raw,truth_sha256=expected_truths[rank])
        for role in ('bank','null'):
            row['adjusted_wins'][role] = row['per_role']['terminal']['adjusted'] < row['per_role'][role]['adjusted']
            row['raw_wins'][role] = native_raw < row['per_role'][role]['raw_mse']
        rows.append(row)
        print(json.dumps(dict(rank=rank,roles={k:{q:v[q] for q in ('raw_mse','adjusted','flops')}
                                             for k,v in row['per_role'].items()})),flush=True)
    summaries = {role:dict(mean_raw_mse=math.fsum(r['per_role'][role]['raw_mse'] for r in rows)/40,
        mean_adjusted=math.fsum(r['per_role'][role]['adjusted'] for r in rows)/40,
        mean_complete_flops=sum(r['per_role'][role]['flops'] for r in rows)/40,
        initialization_flops=acquisitions[role]['initialization_flops']) for role in acquisitions}
    comparisons = {}
    for role in ('bank','null'):
        ratio = summaries['terminal']['mean_adjusted']/summaries[role]['mean_adjusted']
        wins = sum(r['adjusted_wins'][role] for r in rows)
        comparisons[role] = dict(mean_adjusted_ratio=ratio,
            mean_raw_ratio=summaries['terminal']['mean_raw_mse']/summaries[role]['mean_raw_mse'],
            adjusted_wins=wins,raw_wins=sum(r['raw_wins'][role] for r in rows),
            passed=ratio < 1 and wins >= 24)
    unchanged(pins); unchanged(truth_pins)
    require(sha(out/'predictions_and_bills_sealed.json') == before, 'Pre-truth seal unchanged')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2e9, 'Score RSS<2GB')
    result.update(passed=True,completed=True,integrity_passed=True,records=rows,summary=summaries,
        comparisons=comparisons,advances_to_independent14=all(v['passed'] for v in comparisons.values()),
        truth_bindings=truth_pins,hashes_unchanged=True,source_bindings=pins,
        aggregation='Mean per-net observed final MSE * max(.1, own complete F/2^41); initialization once; repeat excluded',
        frame='Reused public-development VALID40 existing1e9 mc_post means, not independent14 or official.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge',action='store_true')
    for name in ('source','builder','native-source','training','build','null','terminal'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.judge:
        parser.print_help();return 0
    require(all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
                for k,v in vars(args).items() if k!='judge'),'All reviewed source and result seals required')
    require(Path(sys.prefix).resolve() == (P2/'data_gate/.venv').resolve() and
            importlib.metadata.version('numpy') == '2.4.6','Pinned OFFPY judge')
    require(os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k)=='1' for k in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'nice10 BLAS1')
    os.chdir(NEW)
    out = HERE/'terminal_score_v1'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(schema='current_paircap4096_e2e_terminal_native40_v1',passed=False,completed=False,
        integrity_passed=False,truth_accessed=False,started_utc=utc(),requested_seals=vars(args),
        official_score=False,production_qualified=False,refit_performed=False)
    write(out/'started.json',result)
    def expired(_signum,_frame):
        raise TimeoutError('Fixed300-second score watchdog')
    signal.signal(signal.SIGALRM,expired);signal.alarm(300)
    try:
        judge(args,out,result)
    except BaseException:
        result['passed']=result['completed']=False
        result['error']=traceback.format_exc()
    finally:
        signal.alarm(0)
        result.update(finished_utc=utc(),wall_s=time.monotonic()-started,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','advances_to_independent14','summary','comparisons','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
