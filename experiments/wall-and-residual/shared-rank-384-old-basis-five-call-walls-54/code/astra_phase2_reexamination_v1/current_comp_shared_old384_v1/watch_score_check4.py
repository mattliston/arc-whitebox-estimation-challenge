"""Inert inverse of the completed lightweight score watchdog; fixed CHECK4."""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OWNER = HERE/'watch_score_probe.py'
OWNER_SHA = 'ce125eeb2bf774f9647b925018ba3ea1bf4e7b0a4e18277d6fab2d1c09d7d6a5'
RANKS = (37,39,42,48)


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def derive_source():
    original = OWNER.read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == OWNER_SHA,'Completed score watch owner')
    edits = [
        ("HERE/'watch_score_probe.py'","HERE/'watch_score_check4.py'",2),
        ("HERE/'score_probe.py'","HERE/'score_check4.py'",4),
        ("HERE/'SCORE_PROBE_PROTOCOL.md'","HERE/'SCORE_CHECK4_PROTOCOL.md'",2),
        ('TRANSPORT:TRANSPORT_SHA,**h.MON_PINS','TRANSPORT:TRANSPORT_SHA,OWNER:OWNER_SHA,**h.MON_PINS',1),
        ("HERE/'probe_v1/result.json'","HERE/'check4_v1/result.json'",1),
        ("HERE/'probe_watch_v1/result.json'","HERE/'check4_watch_v1/result.json'",1),
        ("HERE/'probe.py'","HERE/'check4.py'",1),
        ("HERE/'watch_probe.py'","HERE/'watch_check4.py'",1),
        ("HERE/'PROBE_PROTOCOL.md'","HERE/'CHECK4_PROTOCOL.md'",1),
        ("HERE/'PROBE_WATCH_SPEC.md'","HERE/'CHECK4_WATCH_SPEC.md'",1),
        ("HERE/'score_probe_v1',HERE/'score_probe_watch_v1'","HERE/'score_check4_v1',HERE/'score_check4_watch_v1'",1),
        ("schema='current_comp_shared_old384_score_probe_watch_v1'","schema='current_comp_shared_old384_score_check4_watch_v1'",1),
        ('all_four_outputs_verified','all_ten_outputs_verified',2),
        ("r['ranks'] == [37]","r['ranks'] == list(RANKS)",1),
        ("(r['comparison']['adjusted_ratio'] <= .95)",
         "(r['comparison']['adjusted_ratio'] <= .97 and r['comparison']['adjusted_wins'] >= 3 and r['comparison']['maximum_adjusted_ratio'] <= 1.10)",1),
        ("r['truth_bindings'] == {str(scorer.TRUTH):scorer.TRUTH_SHA}",
         "r['truth_bindings'] == {str(p):h for p,h in scorer.TRUTHS.items()}",1),
    ]
    changed = original
    for old,new,count in edits:
        require(changed.count(old) == count,'Exact CHECK4 score-watch span '+repr((old,count,changed.count(old))))
        changed = changed.replace(old,new)
    inverse = changed
    for old,new,count in reversed(edits):
        require(inverse.count(new) == count,'Exact CHECK4 score-watch inverse '+new)
        inverse = inverse.replace(new,old)
    require(inverse == original,'Whole lightweight-watch inverse; identical resource loop')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_120s_1GB_no_swap_1s_completed_observations=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert; explicit --run and reviewed source/acquisition hashes required.')
    source,proof = derive_source()
    exec(compile(source,str(OWNER)+':fixed-check4-inverse','exec'),globals())
