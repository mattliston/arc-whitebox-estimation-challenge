"""One fixed offline CHECK4 scorer; exact bounded-supervisor source adapter."""
import hashlib
import json
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_receiver_birth_v1/watch_qualification.py'
_old_sha = '4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb'
_forwarded = ('protocol', 'native-source', 'native-watch-source', 'native-protocol',
    'builder', 'build-protocol') + tuple(arm+'-'+kind for arm in ('p-only', 'joint')
    for kind in ('build', 'candidate', 'native', 'watch'))


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _score_pins(here, args):
    paths = {'protocol': here/'SCORE_PROTOCOL.md', 'native-source': here/'native.py',
        'native-watch-source': here/'watch_native.py', 'native-protocol': here/'NATIVE_PROTOCOL.md',
        'builder': here/'build.py', 'build-protocol': here/'BUILD_PROTOCOL.md'}
    for arm in ('p_only', 'joint'):
        prefix = arm.replace('_', '-')
        paths.update({prefix+'-build': here/f'build_{arm}_v1/result.json',
            prefix+'-candidate': here/f'build_{arm}_v1/estimator.py',
            prefix+'-native': here/f'native_{arm}_v1/result.json',
            prefix+'-watch': here/f'native_{arm}_watch_v1/result.json'})
    _require(set(paths) == set(_forwarded), 'Every fixed scorer argument has one actual file')
    return {path: getattr(args, name.replace('-', '_')+'_sha256') for name, path in paths.items()}


def _score_contract(d, args, here):
    expected = dict(run=True, source_sha256=args.worker_sha256)
    expected.update({name.replace('-', '_')+'_sha256': getattr(args, name.replace('-', '_')+'_sha256')
        for name in _forwarded})
    _require(d['schema'] == 'current_comp_joint_p_node_score_v1' and d['stage'] == 'score'
        and d['requested_seals'] == expected, 'Exact fixed scorer and all15 supplied bindings')
    _require(all(d.get(k) is True for k in ('passed', 'completed', 'integrity_passed',
        'hashes_unchanged', 'sources_unchanged', 'all_predictions_verified_before_truth',
        'exact_COMP_score_replay', 'repeat_excluded', 'no_native_gradient_claim')),
        'Complete authenticated score, not a partial or surrogate result')
    _require(all(d.get(k) is False for k in ('estimator_executed', 'training_or_refit_performed',
        'MPS_score_used', 'automatic_next_stage', 'official_score', 'production_qualified',
        'independent_panel_qualified')), 'No model/refit/MPS/release or automatic action')
    _require(d['ranks'] == [37, 39, 42, 48]
        and [row['rank'] for row in d['records']] == [37, 39, 42, 48]
        and d['elapsed_s'] < 120 and 0 < d['peak_rss_bytes'] < 1_000_000_000,
        'All four fixed results within own offline bounds')
    for path, digest in {here/'score.py': args.worker_sha256, **_score_pins(here, args)}.items():
        _require(d['input_bindings'].get(str(path)) == digest, 'Scorer owns every reviewed source/result/candidate')
    seal = here/'score_v1/predictions_and_bills_sealed.json'
    _require(_sha(seal) == d['predictions_and_bills_seal_sha256'] == d['artifacts'][str(seal)],
        'Same pretruth output and complete-bill seal')
    value = json.loads(seal.read_text())
    _require(value['truth_accessed'] is False and value['ranks'] == [37, 39, 42, 48]
        and all(value.get(k) is True for k in ('all_ten_new_outputs_verified',
            'all_native_bills_and_inputs_verified', 'repeat_excluded', 'own_complete_bills')),
        'All10 outputs and own bills before any truth')


_require(_sha(_parent) == _old_sha, 'Frozen original supervisor changed')
_original = _parent.read_text()
_changes = (
    ('"""Bounded one-MPS qualification supervisor. No numerical imports."""',
     '"""Bounded one-CPU offline scorer. No numerical imports."""'),
    ("AP='/Users/matt/anaconda3/bin/python'", "AP=str(P2/'data_gate/.venv/bin/python')"),
    ("ap.add_argument('--mps-slot-clear',action='store_true')", "ap.add_argument('--cpu-slot-clear',action='store_true')"),
    ("for k in ('source','worker','model','protocol'):ap.add_argument('--'+k+'-sha256')",
     "for k in ('source','worker')+SCORE_FORWARDED:ap.add_argument('--'+k+'-sha256')"),
    ("check(args.mps_slot_clear,'Explicit MPS slot attestation')", "check(args.cpu_slot_clear,'Explicit sole CPU slot attestation')"),
    ("pins={Path(__file__).resolve():args.source_sha256,HERE/'qualification.py':args.worker_sha256,\n"
     "          HERE/'model.py':args.model_sha256,HERE/'PROTOCOL.md':args.protocol_sha256,\n"
     "          HERE/'WATCH_SPEC.md':sha(HERE/'WATCH_SPEC.md'),MON:MON_SHA,P1W:P1W_SHA}",
     "pins={Path(__file__).resolve():args.source_sha256,HERE/'score.py':args.worker_sha256,\n"
     "          HERE/'SCORE_WATCH_SPEC.md':sha(HERE/'SCORE_WATCH_SPEC.md'),MON:MON_SHA,P1W:P1W_SHA,\n"
     "          HERE.parent/'current_comp_receiver_birth_v1/watch_qualification.py':SCORE_OWNER_SHA,\n"
     "          **score_pins(HERE,args)}"),
    ("HERE/'qualification_watch_v1'", "HERE/'score_watch_v1'"),
    ("result=dict(passed=False,source_bindings={str(p):h for p,h in pins.items()},\n"
     "                mps_slot_clear_attested=True,numerical_imports=False,whole_cap_s=180,",
     "result=dict(schema='current_comp_joint_p_node_score_watch_v1',stage='score',passed=False,completed=False,\n"
     "                source_bindings={str(p):h for p,h in pins.items()},input_bindings={str(p):h for p,h in pins.items()},\n"
     "                cpu_slot_clear_attested=True,numerical_imports=False,full_model_calls=0,whole_cap_s=120,"),
    ("check(s['combined_rss_bytes']<12_000_000_000,'12GB combined RSS')",
     "check(s['combined_rss_bytes']<1_000_000_000,'1GB combined RSS')"),
    ("time.monotonic()-began<180", "time.monotonic()-began<120"),
    ("180s whole cap", "120s whole cap"),
    ("        if readlog is not None:\n"
     "            pending+=readlog.read()\n"
     "            while '\\n' in pending:\n"
     "                line,pending=pending.split('\\n',1)\n"
     "                try:item=json.loads(line)\n"
     "                except (ValueError,TypeError):continue\n"
     "                if isinstance(item,dict) and item.get('event')=='retired':\n"
     "                    check(time.monotonic()-last_progress<120,'Prior retired-call deadline already expired')\n"
     "                    check(item['call']==result['retired_calls'],'Ordered retired call marker')\n"
     "                    result['retired_calls']+=1;last_progress=time.monotonic()\n", ""),
    ("'120s without fully retired call'", "'120s absolute offline-score completion deadline'"),
    ("argv=[AP,str(HERE/'qualification.py'),'--run','--source-sha256',args.worker_sha256,\n"
     "                  '--model-sha256',args.model_sha256,'--protocol-sha256',args.protocol_sha256,\n"
     "                  '--watcher-sha256',args.source_sha256]",
     "argv=[AP,str(HERE/'score.py'),'--run','--source-sha256',args.worker_sha256]\n"
     "            for name in SCORE_FORWARDED:\n"
     "                argv.extend(['--'+name+'-sha256',getattr(args,name.replace('-','_')+'_sha256')])"),
    ("log=(out/'worker.log').open('x');readlog=(out/'worker.log').open('r')", "log=(out/'worker.log').open('x')"),
    ("HERE/'qualification_v1/result.json'", "HERE/'score_v1/result.json'"),
    ("d['passed'] and d['completed'] and c['passed'] and c['result_sha256']", "d['passed'] and d['completed'] and c['passed'] and c['completed'] and c['result_sha256']"),
    ("f['elapsed_s']<180", "f['elapsed_s']<120"),
    ("check(result['retired_calls']==d['retired_call_count']==8\n"
     "                  and d['full_model_calls']==7 and d['tiny_model_calls']==8,'Complete fixed qualification roster')",
     "check(result['retired_calls']==0,'No model-call marker or model work')\n"
     "            score_contract(d,args,HERE)"),
    ("result.update(passed=True,child_result_sha256=sha(record)", "result.update(passed=True,completed=True,child_result_sha256=sha(record)"),
    ("result.update(passed=False,error=f'{type(e).__name__}: {e}')", "result.update(passed=False,completed=False,error=f'{type(e).__name__}: {e}')"),
    ("result['elapsed_s']>=180", "result['elapsed_s']>=120"),
    ("sealed_elapsed<180", "sealed_elapsed<120"),
    ("final_elapsed<180", "final_elapsed<120"),
)
_derived, _counts = _original, []
for _before, _after in _changes:
    _count = _derived.count(_before)
    _require(_count > 0, 'Missing reviewed supervisor span: '+_before)
    _counts.append(_count)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for (_before, _after), _count in reversed(list(zip(_changes, _counts))):
    if not _after:
        # The removed progress parser had one explicit stable neighboring span.
        _anchor = "        check(time.monotonic()-last_progress<120,'120s without fully retired call')\n"
        _require(_count == 1 and _inverse.count(_anchor) == 1, 'Unique removed model-progress inverse')
        _inverse = _inverse.replace(_anchor, _before+_anchor)
    else:
        _require(_inverse.count(_after) == _count, 'Unexpected inverse supervisor span: '+_after)
        _inverse = _inverse.replace(_after, _before)
_require(_inverse == _original, 'Exact original supervisor restored')
exec(compile(_derived, str(_parent)+':joint-fixed-offline-score-v1', 'exec'),
    {'__name__': __name__, '__file__': __file__, 'SCORE_FORWARDED': _forwarded,
     'SCORE_OWNER_SHA': _old_sha, 'score_pins': _score_pins, 'score_contract': _score_contract})
