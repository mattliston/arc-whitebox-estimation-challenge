"""Inert v2 scalar supervisor: same actual monitor, corrected metadata/tail scope."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE/'watch_scalar_semantics.py'
OWNER_SHA = '721667394b12ea6292fa66b045eff1aefb642af08e9606b6da4951a4d39cd4f9'
OLD_SPEC = HERE/'SCALAR_SEMANTICS_WATCH_SPEC.md'
OLD_SPEC_SHA = '3c1a8dd97512813c31d2e1a244169272933cb5c93b097641ebaa9c682be9109a'


def derive_source():
    raw = OWNER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OWNER_SHA or hashlib.sha256(OLD_SPEC.read_bytes()).hexdigest() != OLD_SPEC_SHA:
        raise RuntimeError('Frozen original scalar supervisor and scope')
    original = raw.decode()
    before = """        if result['passed']:
            observe()
        result['elapsed_s'] = time.monotonic()-began
"""
    after = """        try:
            if result['passed']:
                observe()
        except BaseException:
            result.update(passed=False,completed=False,final_observation_error=traceback.format_exc())
        stream.close()
        publication_deadline = min(previous+1.,began+CAP)
        for path in (out/'started.json',out/'samples.jsonl',out/'local.log',out/'server.log',out/'remote.log'):
            if path.exists():
                result['artifacts'][str(path)] = sha(path)
        if result['passed']:
            require(time.monotonic() < publication_deadline,'Original completed-observation publication deadline')
        result['no_post_publication_observation_claimed'] = True
        result['publication_deadline_monotonic_s'] = publication_deadline
        result['elapsed_s'] = time.monotonic()-began
"""
    old_tail = """        if ok:
            observe(); deadline()
        stream.close()
"""
    new_tail = """        if ok:
            require(time.monotonic() < publication_deadline,'Final write inside original1s/60s tail')
            deadline()
"""
    edits = [
        ("worker = HERE/'scalar_semantics_probe.py'", "worker = HERE/'scalar_semantics_probe_v2.py'"),
        ("HERE/'SCALAR_SEMANTICS_PROTOCOL.md'", "HERE/'SCALAR_SEMANTICS_PROTOCOL_V2.md'"),
        ("HERE/'SCALAR_SEMANTICS_WATCH_SPEC.md'", "HERE/'SCALAR_SEMANTICS_WATCH_V2_SPEC.md'"),
        ("out,root = HERE/'scalar_semantics_watch_v1',HERE/'scalar_semantics_v1'",
         "out,root = HERE/'scalar_semantics_watch_v2',HERE/'scalar_semantics_v2'"),
        ("schema='current_comp_shared_old384_scalar_semantics_watch_v1'",
         "schema='current_comp_shared_old384_scalar_semantics_watch_v2'"),
        ("r['schema'] == 'current_comp_shared_old384_scalar_semantics_v1'",
         "r['schema'] == 'current_comp_shared_old384_scalar_semantics_v2'"),
        ("HERE/'SCALAR_SEMANTICS_WATCH_V2_SPEC.md':args.spec_sha256,TRANSPORT:TRANSPORT_SHA}",
         "HERE/'SCALAR_SEMANTICS_WATCH_V2_SPEC.md':args.spec_sha256,TRANSPORT:TRANSPORT_SHA,\n"
         "        OWNER:OWNER_SHA,OLD_SPEC:OLD_SPEC_SHA}"),
        (before,after),(old_tail,new_tail)]
    changed = original
    for old,new in edits:
        if changed.count(old) != 1:
            raise RuntimeError('Unique watch v2 metadata/tail edit: '+old[:70])
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        if inverse.count(new) != 1:
            raise RuntimeError('Unique watch v2 inverse: '+new[:70])
        inverse = inverse.replace(new,old)
    if inverse.encode() != raw:
        raise RuntimeError('Whole original supervisor inverse')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        numerical_question_and_resource_loop_unchanged=True,bounded_sealed_publication_tail=True)


_source,SOURCE_ADAPTER_PROOF = derive_source()
_api = dict(__file__=str(Path(__file__).resolve()),__name__='scalar_watch_v2_inert',
            OWNER=OWNER,OWNER_SHA=OWNER_SHA,OLD_SPEC=OLD_SPEC,OLD_SPEC_SHA=OLD_SPEC_SHA)
exec(compile(_source,str(Path(__file__).resolve())+':v2-source-inverse','exec'),_api)
main = _api['main']
del _source


if __name__ == '__main__':
    raise SystemExit(main())
