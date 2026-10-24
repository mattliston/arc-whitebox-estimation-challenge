"""Original120s/2GB independent14 score monitor, current source authorities."""
import hashlib
from pathlib import Path
import runpy
import sys

_GSW_HERE = Path(__file__).resolve().parent
_GSW_SHARED = _GSW_HERE.parent/'current_comp_shared_old384_v1'
_GSW_OWNER = _GSW_SHARED/'watch_score_panel_v2.py'
_GSW_OWNER_SHA = '6500fad9fffd81a82bd5fa4861210ad43d35c6d65b45bbf498dc153087a4e223'
_GSW_PINS = {_GSW_OWNER:_GSW_OWNER_SHA}
_OWNER = _GSW_SHARED/'watch_score_probe.py'
_OWNER_SHA = 'ce125eeb2bf774f9647b925018ba3ea1bf4e7b0a4e18277d6fab2d1c09d7d6a5'


def derive_source():
    if _GSW_OWNER.is_symlink() or hashlib.sha256(_GSW_OWNER.read_bytes()).hexdigest() != _GSW_OWNER_SHA:
        raise RuntimeError('Exact completed independent14 score-watch source')
    parent=runpy.run_path(str(_GSW_OWNER),run_name='_inert_gain_score_watch_parent')
    original,prior=parent['derive_source']()
    edits=(
        ("        'native40','native40-watch','score40','score40-watch')",
         "        'native40','native40-watch','score40','score40-watch','build','pair')",1),
        ("HERE/'score_panel_v2.py'","HERE/'score_panel.py'",3),
        ("HERE/'watch_score_panel_v2.py'","HERE/'watch_score_panel.py'",1),
        ("HERE/'SCORE_PANEL_V2_PROTOCOL.md'","HERE/'SCORE_PANEL_PROTOCOL.md'",1),
        ("HERE/'SCORE_PANEL_V2_WATCH_SPEC.md'","HERE/'SCORE_PANEL_WATCH_SPEC.md'",1),
        ("'current_comp_shared_old384_score_panel_watch_v2'","'current_comp_shared_gain_slice_score_panel_watch_v1'",1),
        ('own.update({TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA,**h.MON_PINS})',
         'own.update({TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA,**_GSW_PINS,**h.MON_PINS})',1),
    )
    changed,spans=original,[]
    for before,after,count in edits:
        if changed.count(before)!=count:
            raise RuntimeError('Exact gain score-watch source boundary: '+before[:100])
        cursor=0
        for unused in range(count):
            pos=changed.index(before,cursor);spans.append((pos,before,after))
            changed=changed[:pos]+after+changed[pos+len(before):];cursor=pos+len(after)
    inverse=changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)]!=after:
            raise RuntimeError('Positional gain score-watch inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire positive score supervisor loop restored byte-exact')
    return changed,dict(parent=prior,owner_sha256=_GSW_OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),source_splices=len(spans),
        original_sampling_reaping_finalization_and_truth_byte_allowlist_unchanged=True,
        whole_s=120,combined_rss_bytes=2_000_000_000,current_backend_and_historical_comparator_policy_required=True)


if __name__=='__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert; explicit --run and reviewed current source/actual result hashes required.')
    source,proof=derive_source()
    exec(compile(source,str(_GSW_OWNER)+':gain-panel-score-watch','exec'),globals())
