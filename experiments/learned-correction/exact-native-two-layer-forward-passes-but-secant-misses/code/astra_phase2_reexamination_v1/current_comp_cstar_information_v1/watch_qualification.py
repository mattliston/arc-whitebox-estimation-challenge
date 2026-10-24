"""Fresh fixed-state cstar qualification supervisor; stdlib only on import."""
import hashlib
import math
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_receiver_birth_v1/watch_qualification.py'
_parent_sha = '4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb'
_protocol_sha = 'c3660d88e1a9e90d3ff040b5f9cf191a323b0cf00589d1167fe4ff60bf851682'
_flags = ('execution_integrity_passed', 'numerical_qualification_passed',
    'all_graphs_retired', 'all16_domains_qualified', 'all47_selectors_qualified',
    'same_backend_null_byte_exact', 'repeat_gradient_byte_exact',
    'strict_secant_gates_passed', 'strict_gradient_gates_passed',
    'served_vector_ad_identity_passed', 'frozen_P_A_cstar_weights_passed',
    'hashes_unchanged', 'sources_unchanged')


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _marker(item, call):
    _require(call < 34 and item.get('device') == ('cpu' if call < 17 else 'mps')
        and item.get('rank') in (35, 100) and isinstance(item.get('kind'), str),
        'Only the prescribed CPU-then-MPS qualification roster')
    driver, host = item.get('mps_driver_bytes'), item.get('host_peak_rss_bytes')
    _require(isinstance(driver, (int, float)) and math.isfinite(driver)
        and 0 <= driver < 16_000_000_000 and isinstance(host, (int, float))
        and math.isfinite(host) and 0 < host < 12_000_000_000,
        'Every retired call reports bounded actual driver and host observations')


def _contract(record):
    _require(record.get('schema') == 'current_comp_cstar_qualification_v1'
        and record.get('stage') == 'qualification'
        and all(record.get(k) is True for k in _flags)
        and record.get('native_gradient_qualified') is False,
        'Fresh complete controls, all original gates, no native-gradient claim')
    for group in ('source_bindings', 'input_bindings', 'artifacts'):
        _require(isinstance(record.get(group), dict) and record[group],
            'Complete source/input/evidence ownership: '+group)


_require(hashlib.sha256(_parent.read_bytes()).hexdigest() == _parent_sha,
    'Frozen original independent qualification supervisor')
_original = _parent.read_text()
_changes = (
    ("for k in ('source','worker','model','protocol'):ap.add_argument('--'+k+'-sha256')",
     "for k in ('source','worker','model','protocol','spec'):ap.add_argument('--'+k+'-sha256')", 1),
    ("HERE/'model.py':args.model_sha256,HERE/'PROTOCOL.md':args.protocol_sha256,",
     "HERE/'model.py':args.model_sha256,HERE/'QUALIFICATION_PROTOCOL.md':args.protocol_sha256,", 1),
    ("HERE/'WATCH_SPEC.md':sha(HERE/'WATCH_SPEC.md'),MON:MON_SHA,P1W:P1W_SHA}",
     "HERE/'WATCH_SPEC.md':args.spec_sha256,MON:MON_SHA,P1W:P1W_SHA,\n"
     "          CSTAR_PARENT:CSTAR_PARENT_SHA,HERE/'PROTOCOL.md':CSTAR_PROTOCOL_SHA}", 1),
    ("result=dict(passed=False,source_bindings={str(p):h for p,h in pins.items()},",
     "result=dict(schema='current_comp_cstar_qualification_watch_v1',stage='qualification',\n"
     "                passed=False,source_bindings={str(p):h for p,h in pins.items()},", 1),
    ('180', '600', 8),
    ("                    check(item['call']==result['retired_calls'],'Ordered retired call marker')\n",
     "                    check(item['call']==result['retired_calls'],'Ordered retired call marker')\n"
     "                    cstar_marker(item,result['retired_calls'])\n", 1),
    ("'--watcher-sha256',args.source_sha256]",
     "'--watcher-sha256',args.source_sha256,'--watch-spec-sha256',args.spec_sha256]", 1),
    ("check(result['retired_calls']==d['retired_call_count']==8\n"
     "                  and d['full_model_calls']==7 and d['tiny_model_calls']==8,'Complete fixed qualification roster')",
     "check(result['retired_calls']==d['retired_call_count']==34\n"
     "                  and d['full_model_calls']==34 and d['tiny_model_calls']==0,'Complete fixed34-call qualification roster')\n"
     "            cstar_contract(d)", 1),
)
_derived = _original
for _before, _after, _count in _changes:
    _require(_derived.count(_before) == _count, 'Exact cstar supervisor source span: '+_before)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after, _count in reversed(_changes):
    _require(_inverse.count(_after) == _count, 'Exact cstar supervisor inverse span: '+_after)
    _inverse = _inverse.replace(_after, _before)
_require(_inverse == _original, 'Entire original independent supervisor restored')
_proof = dict(parent=str(_parent), parent_sha256=_parent_sha,
    full_source_inverse_verified=True, absolute_monitor_and_cleanup_unchanged=True,
    fixed34_call_CPU_then_MPS_roster=True, whole600s_fresh_protocol=True,
    old_failed_qualifications_not_reinterpreted=True)
exec(compile(_derived, str(_parent)+':fresh-cstar-fixed-state', 'exec'),
    {'__name__': __name__, '__file__': __file__, 'CSTAR_PARENT': _parent,
     'CSTAR_PARENT_SHA': _parent_sha, 'CSTAR_PROTOCOL_SHA': _protocol_sha,
     'cstar_marker': _marker, 'cstar_contract': _contract})
