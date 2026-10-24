"""Identical bounded CHECK4 score supervisor, fresh v2 path metadata only."""
import hashlib
from pathlib import Path
import runpy
import sys

_V2_HERE = Path(__file__).resolve().parent
_V2_OWNER = _V2_HERE/'watch_score_check4.py'
_V2_OWNER_SHA = '35e1c07bada703fc26057ab482f844daeef29018fbb907cd03a77311c24e0bf7'


def derive_source():
    if hashlib.sha256(_V2_OWNER.read_bytes()).hexdigest() != _V2_OWNER_SHA:
        raise RuntimeError('Frozen unexecuted CHECK4 score watch owner')
    parent = runpy.run_path(str(_V2_OWNER),run_name='inert_check4_score_watch_v1_owner')
    original,parent_proof = parent['derive_source']()
    # Required inert variables of the first version's derived source.
    globals().update({k:parent[k] for k in ('OWNER','OWNER_SHA','RANKS')})
    edits = [
        ("HERE/'watch_score_check4.py'", "HERE/'watch_score_check4_v2.py'", 2),
        ("HERE/'score_check4.py'", "HERE/'score_check4_v2.py'", 4),
        ("HERE/'SCORE_CHECK4_PROTOCOL.md'", "HERE/'SCORE_CHECK4_V2_PROTOCOL.md'", 2),
        ("HERE/'check4_v1/result.json'", "HERE/'check4_v2/result.json'", 1),
        ("HERE/'check4_watch_v1/result.json'", "HERE/'check4_watch_v2/result.json'", 1),
        ("HERE/'check4.py'", "HERE/'check4_v2.py'", 1),
        ("HERE/'watch_check4.py'", "HERE/'watch_check4_v2.py'", 1),
        ("HERE/'CHECK4_PROTOCOL.md'", "HERE/'CHECK4_V2_PROTOCOL.md'", 1),
        ("HERE/'CHECK4_WATCH_SPEC.md'", "HERE/'CHECK4_V2_WATCH_SPEC.md'", 1),
        ("HERE/'score_check4_v1',HERE/'score_check4_watch_v1'", "HERE/'score_check4_v2',HERE/'score_check4_watch_v2'", 1),
        ("schema='current_comp_shared_old384_score_check4_watch_v1'", "schema='current_comp_shared_old384_score_check4_watch_v2'", 1),
        ('OWNER:OWNER_SHA,**h.MON_PINS', 'OWNER:OWNER_SHA,_V2_OWNER:_V2_OWNER_SHA,**h.MON_PINS', 1),
    ]
    changed = original
    for before,after,count in edits:
        if changed.count(before) != count:
            raise RuntimeError('Unique score-watch-v2 span '+repr((before,changed.count(before),count)))
        changed = changed.replace(before,after)
    inverse = changed
    for before,after,count in reversed(edits):
        if inverse.count(after) != count:
            raise RuntimeError('Unique score-watch-v2 inverse '+after)
        inverse = inverse.replace(after,before)
    if inverse != original:
        raise RuntimeError('Entire original CHECK4 score watcher restored')
    return changed,dict(owner_sha256=_V2_OWNER_SHA,parent=parent_proof,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_120s_1GB_no_swap_1s_completed_observations=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert; explicit --run and reviewed source/acquisition hashes required.')
    source,proof = derive_source()
    exec(compile(source,str(_V2_OWNER)+':fresh-acquisition-metadata-v2','exec'),globals())
