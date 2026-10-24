"""Inert fresh-owner adapter of the complete reviewed v2 scalar supervisor."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_OWNER = HERE/'watch_scalar_semantics_v2.py'
V2_OWNER_SHA = 'f1242f9ee7e382501262d3cf0b7a4b9c89d7de8018b83fccf44d826fe973b31a'
V2_SPEC = HERE/'SCALAR_SEMANTICS_WATCH_V2_SPEC.md'
V2_SPEC_SHA = 'f7b5a0736c4e94934eec1ba9c6900b1feea22666209fdbee307e00b2c2c5a54f'
for _p,_d in ((V2_OWNER,V2_OWNER_SHA),(V2_SPEC,V2_SPEC_SHA)):
    if hashlib.sha256(_p.read_bytes()).hexdigest() != _d:
        raise RuntimeError('Frozen v2 supervisor and publication contract')
_spec = importlib.util.spec_from_file_location('scalar_watch_v3_original_v2',V2_OWNER)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def derive_source():
    original,parent = _base.derive_source()
    edits = [
        ("HERE/'scalar_semantics_probe_v2.py'", "HERE/'scalar_semantics_probe_v3.py'"),
        ("HERE/'SCALAR_SEMANTICS_PROTOCOL_V2.md'", "HERE/'SCALAR_SEMANTICS_PROTOCOL_V3.md'"),
        ("HERE/'SCALAR_SEMANTICS_WATCH_V2_SPEC.md'", "HERE/'SCALAR_SEMANTICS_WATCH_V3_SPEC.md'"),
        ("HERE/'scalar_semantics_watch_v2',HERE/'scalar_semantics_v2'",
         "HERE/'scalar_semantics_watch_v3',HERE/'scalar_semantics_v3'"),
        ("current_comp_shared_old384_scalar_semantics_watch_v2", "current_comp_shared_old384_scalar_semantics_watch_v3"),
        ("current_comp_shared_old384_scalar_semantics_v2", "current_comp_shared_old384_scalar_semantics_v3"),
        ("OWNER:OWNER_SHA,OLD_SPEC:OLD_SPEC_SHA}",
         "OWNER:OWNER_SHA,OLD_SPEC:OLD_SPEC_SHA,V2_OWNER:V2_OWNER_SHA,V2_SPEC:V2_SPEC_SHA}")]
    changed = original
    for old,new in edits:
        if changed.count(old) != 1 or new in changed:
            raise RuntimeError('Unique v3 owner/schema edit: '+old)
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        if inverse.count(new) != 1:
            raise RuntimeError('Unique v3 inverse')
        inverse = inverse.replace(new,old)
    if inverse != original:
        raise RuntimeError('Whole v2 supervisor inverse')
    return changed,dict(parent=parent,owner_sha256=V2_OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original_numerical_question_resources_and_publication_tail_unchanged=True)


_source,SOURCE_ADAPTER_PROOF = derive_source()
_api = dict(_base._api,__file__=str(Path(__file__).resolve()),__name__='scalar_watch_v3_inert',
            V2_OWNER=V2_OWNER,V2_OWNER_SHA=V2_OWNER_SHA,V2_SPEC=V2_SPEC,V2_SPEC_SHA=V2_SPEC_SHA)
exec(compile(_source,str(Path(__file__).resolve())+':fresh-v3-owners','exec'),_api)
main = _api['main']
del _source


if __name__ == '__main__':
    raise SystemExit(main())
