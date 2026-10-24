"""Conditional fixed40 saved-score watch; original complete resource loop."""
import hashlib
from pathlib import Path
import runpy
import sys

_S40_HERE = Path(__file__).resolve().parent
_S40_OWNER = _S40_HERE/'watch_score_check4_v2.py'
_S40_OWNER_SHA = '708731161669da955f3640bc6e0fb0bc4a1d8631da54233dce4fcd30c0b79faa'
_S40_RANKS = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,
              146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234)


def derive_source():
    if hashlib.sha256(_S40_OWNER.read_bytes()).hexdigest() != _S40_OWNER_SHA:
        raise RuntimeError('Fixed CHECK4 v2 score watchdog source')
    parent = runpy.run_path(str(_S40_OWNER),run_name='inert_score40_watch_parent')
    original,parent_proof = parent['derive_source']()
    env = parent['derive_source'].__globals__
    globals().update({k:env[k] for k in ('OWNER','OWNER_SHA','_V2_OWNER','_V2_OWNER_SHA')})
    globals()['RANKS'] = _S40_RANKS
    edits = [
        ("'probe-protocol','probe-watch-spec','probe','probe-watch')", "'probe-protocol','probe-watch-spec','probe','probe-watch','check4-score','check4-watch')", 1),
        ("HERE/'watch_score_check4_v2.py'", "HERE/'watch_score40.py'", 2),
        ("HERE/'score_check4_v2.py'", "HERE/'score40.py'", 4),
        ("HERE/'SCORE_CHECK4_V2_PROTOCOL.md'", "HERE/'SCORE40_PROTOCOL.md'", 2),
        ("HERE/'check4_v2/result.json'", "HERE/'native40_v1/result.json'", 1),
        ("HERE/'check4_watch_v2/result.json'", "HERE/'native40_watch_v1/result.json'", 1),
        ("HERE/'check4_v2.py'", "HERE/'native40.py'", 1),
        ("HERE/'watch_check4_v2.py'", "HERE/'watch_native40.py'", 1),
        ("HERE/'CHECK4_V2_PROTOCOL.md'", "HERE/'NATIVE40_PROTOCOL.md'", 1),
        ("HERE/'CHECK4_V2_WATCH_SPEC.md'", "HERE/'NATIVE40_WATCH_SPEC.md'", 1),
        ("HERE/'score_check4_v2',HERE/'score_check4_watch_v2'", "HERE/'score40_v1',HERE/'score40_watch_v1'", 1),
        ("schema='current_comp_shared_old384_score_check4_watch_v2'", "schema='current_comp_shared_old384_score40_watch_v1'", 1),
        ('_V2_OWNER:_V2_OWNER_SHA,**h.MON_PINS', '_V2_OWNER:_V2_OWNER_SHA,_S40_OWNER:_S40_OWNER_SHA,**h.MON_PINS', 1),
        ("HERE/'NATIVE40_WATCH_SPEC.md':args.probe_watch_spec_sha256}",
         "HERE/'NATIVE40_WATCH_SPEC.md':args.probe_watch_spec_sha256,\n"
         "        HERE/'score_check4_v2/result.json':args.check4_score_sha256,\n"
         "        HERE/'score_check4_watch_v2/result.json':args.check4_watch_sha256}", 1),
        ('all_ten_outputs_verified', 'all_82_outputs_verified', 2),
        ("(r['comparison']['adjusted_ratio'] <= .97 and r['comparison']['adjusted_wins'] >= 3 and r['comparison']['maximum_adjusted_ratio'] <= 1.10)", "(r['comparison']['adjusted_ratio'] <= .97)", 1),
    ]
    changed = original
    for before,after,count in edits:
        if changed.count(before) != count:
            raise RuntimeError('Unique40 score watch span '+repr((before,changed.count(before),count)))
        changed = changed.replace(before,after)
    inverse = changed
    for before,after,count in reversed(edits):
        if inverse.count(after) != count:
            raise RuntimeError('Unique40 score watch inverse '+after)
        inverse = inverse.replace(after,before)
    if inverse != original:
        raise RuntimeError('Entire original CHECK4 v2 score watcher restored')
    return changed,dict(owner_sha256=_S40_OWNER_SHA,parent=parent_proof,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_120s_1GB_no_swap_1s_completed_observations=True,ranks=list(_S40_RANKS))


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert; explicit --run and reviewed source/acquisition/gate hashes required.')
    source,proof = derive_source()
    exec(compile(source,str(_S40_OWNER)+':fixed40-score-watch','exec'),globals())
