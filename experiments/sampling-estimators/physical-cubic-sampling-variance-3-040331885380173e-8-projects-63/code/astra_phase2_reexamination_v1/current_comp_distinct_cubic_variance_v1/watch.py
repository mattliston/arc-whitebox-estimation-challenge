"""DRAFT full NOISE single-RPC watch inverse; new variance meaning only.

No resource, process, retirement, publication or context gate is relaxed.
The empirical variance decision never determines operational PASS.
"""
import hashlib
from pathlib import Path

_HERE=Path(__file__).resolve().parent
_NOISE=_HERE.parent/'current_comp_cubic_innovation_noise_v1'
_OWNER=_NOISE/'watch.py'
_OWNER_SHA='12dee2bcbcc97ddb9382f6b31287309252a0d264ccd3bc14f3e351583c2d25a5'
_SPEC_SHA='cb86135b168c08ff52dff63959b401a0c1b220d8f9db24f44878f1abff599835'


def source_and_proof():
    for path,digest in ((_OWNER,_OWNER_SHA),(_NOISE/'WATCH_SPEC.md',_SPEC_SHA)):
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Exact whole NOISE resource lifecycle source')
    original=_OWNER.read_text();source=original;changes=[]
    def change(before,after,count=1):
        nonlocal source
        if source.count(before)!=count:
            raise RuntimeError('Exact new variance watch splice: '+before[:100])
        for unused in range(count):
            at=source.index(before);changes.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    for before,after in (
        ('current_comp_cubic_innovation_noise','current_comp_distinct_cubic_variance'),
        ('noise_watch_v1','variance_watch_v1'),('noise_v1','variance_v1'),
        ('astra_cubic_noise_','astra_distinct_variance_'),
        ('fixed_noise_only','fixed_distinct_variance_only'),
        ('actual_noise_circuit_qualified','actual_variance_circuit_qualified'),
        ('host_noise_precision_passed','variance_planning_passed'),
        ('        "            outer_variance_measured=False,signed_innovation_measured=False,")',
         '        "            variance_measured=True,signed_current_tensor_error_measured=False,\\n"\n'
         '        "            full_scope_projection_decision_pending=True,")'),
        ('precision_gate_does_not_control_operational_pass=True,numerical_execution=False)',
         'precision_gate_does_not_control_operational_pass=True,\n'
         '        complete_NOISE_watch_inverse=_DISTINCT_WATCH_PROOF,numerical_execution=False)'),
    ):
        change(before,after,source.count(before))
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact complete variance watch positional inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire qualified NOISE watcher restored')
    return source,dict(owner_sha256=_OWNER_SHA,owner_spec_sha256=_SPEC_SHA,
        whole_NOISE_source_inverse=True,splice_count=len(changes),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_resource_loop_cleanup_and_sealed_tail_unchanged=True,
        numerical_execution=False)


_derived,_proof=source_and_proof()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_distinct_variance_watch_owner',
          _DISTINCT_WATCH_PROOF=_proof)
exec(compile(_derived,str(_HERE/'watch.py')+':NOISE-source-inverse','exec'),_api)
globals().update({k:v for k,v in _api.items() if not k.startswith('__')})

if __name__=='__main__':
    raise SystemExit(main())
