"""Same fidelity supervisor/resources, versioned saved-score interfaces only."""
import hashlib
from pathlib import Path
import runpy
import sys

_F2W_HERE = Path(__file__).resolve().parent
_F2W_OWNER = _F2W_HERE/'watch_fidelity.py'
_F2W_OWNER_SHA = 'b134f289dc01a09abd7c9f8aec2642fd15d61e6c52ce1a5343f83887f620b62f'


def derive_source():
    if hashlib.sha256(_F2W_OWNER.read_bytes()).hexdigest() != _F2W_OWNER_SHA:
        raise RuntimeError('Frozen original fidelity supervisor')
    parent = runpy.run_path(str(_F2W_OWNER),run_name='inert_original_fidelity_watch')
    original,prior = parent['derive_source']()
    globals().update({k:parent[k] for k in ('_OWNER','_OWNER_SHA')})
    edits = [
        ('watch_fidelity.py','watch_fidelity_v2.py'),
        ('fidelity.py','fidelity_v2.py'),
        ('FIDELITY_PROTOCOL.md','FIDELITY_V2_PROTOCOL.md'),
        ('FIDELITY_WATCH_SPEC.md','FIDELITY_V2_WATCH_SPEC.md'),
        ('fidelity_watch_v1','fidelity_watch_v2'),
        ('_OWNER:_OWNER_SHA,**h.MON_PINS',
         '_OWNER:_OWNER_SHA,_F2W_OWNER:_F2W_OWNER_SHA,**h.MON_PINS'),
    ]
    changed = original; spans = []
    for before,after in edits:
        count = changed.count(before)
        if not count:
            raise RuntimeError('Explicit present fidelity-watch version span '+before)
        positions = []; offset = 0
        for unused in range(count):
            start = changed.index(before,offset);positions.append(start)
            offset = start+len(before)
        for start in reversed(positions):
            changed = changed[:start]+after+changed[start+len(before):]
            spans.append((start,before,after))
    inverse = changed
    for start,before,after in reversed(spans):
        if inverse[start:start+len(after)] != after:
            raise RuntimeError('Exact fidelity-watch inverse cardinality')
        inverse = inverse[:start]+before+inverse[start+len(after):]
    if inverse != original:
        raise RuntimeError('Whole original fidelity resource loop restored')
    return changed,dict(owner_sha256=_F2W_OWNER_SHA,parent=prior,
        whole_source_inverse=True,derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_120s_1GB_zero_swap_1s_completed_samples=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert; explicit reviewed sources and completed positive v2 score required.')
    source,proof = derive_source()
    exec(compile(source,str(_F2W_OWNER)+':versioned-fidelity-watch','exec'),globals())
