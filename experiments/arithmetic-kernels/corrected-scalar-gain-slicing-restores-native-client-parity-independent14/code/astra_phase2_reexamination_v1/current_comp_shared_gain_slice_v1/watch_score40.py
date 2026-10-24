"""Reviewed original120s/1GB saved-score monitor, current ordinary40 authorities."""
import hashlib
from pathlib import Path
import runpy
import sys

_G40SW_HERE=Path(__file__).resolve().parent
_G40SW_OWNER=_G40SW_HERE/'watch_score_panel.py'
_G40SW_OWNER_SHA='3a7fd5a7125aae140d4f4e81d3d7cf24e703620d9917154df43b9014b8cd64cc'
if hashlib.sha256(_G40SW_OWNER.read_bytes()).hexdigest()!=_G40SW_OWNER_SHA:
    raise RuntimeError('Exact completed current saved-score resource monitor')
_G40SW_PARENT=runpy.run_path(str(_G40SW_OWNER),run_name='_inert_current_full40_score_watch_parent')


def derive_source():
    original,prior=_G40SW_PARENT['derive_source']()
    env=_G40SW_PARENT['derive_source'].__globals__
    # Retain the exact monitor transport owner bindings in its actual globals.
    for key in ('_OWNER','_OWNER_SHA','_GSW_PINS'):
        globals()[key]=env[key]
    edits=(
        ("        'native40','native40-watch','score40','score40-watch','build','pair')",
         "        'panel-native','panel-native-watch','panel-score','panel-score-watch','build','pair')",1),
        ('CAP, RSS = 120.,2_000_000_000','CAP, RSS = 120.,1_000_000_000',1),
        ("'Zero added swap and combined owned-tree2GB'","'Zero added swap and combined owned-tree1GB'",1),
        ("HERE/'score_panel.py'","HERE/'score40.py'",3),
        ("HERE/'watch_score_panel.py'","HERE/'watch_score40.py'",1),
        ("HERE/'SCORE_PANEL_PROTOCOL.md'","HERE/'SCORE40_PROTOCOL.md'",1),
        ("HERE/'SCORE_PANEL_WATCH_SPEC.md'","HERE/'SCORE40_WATCH_SPEC.md'",1),
        ("'current_comp_shared_gain_slice_score_panel_watch_v1'",
         "'current_comp_shared_gain_slice_score40_watch_v1'",1),
        ('own.update({TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA,**_GSW_PINS,**h.MON_PINS})',
         'own.update({TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA,**_GSW_PINS,_G40SW_OWNER:_G40SW_OWNER_SHA,**h.MON_PINS})',1),
        ("        for binding in r['truth_bindings'].values():\n"
         "            require(scorer.sha(binding['path'],tick,truth=True) == binding['sha256'],\n"
         "                'Fresh allowlisted original14 truth byte hashes; no supervisor array decoding')",
         "        for path,digest in r['truth_bindings'].items():\n"
         "            require(scorer.sha(path,tick,truth=True) == digest,\n"
         "                'Fresh allowlisted original40 ordinary truth bytes; no supervisor array decoding')",1),
    )
    changed,spans=original,[]
    for before,after,count in edits:
        if changed.count(before)!=count:
            raise RuntimeError('Exact current ordinary40 score-watch boundary: '+before[:100])
        cursor=0
        for unused in range(count):
            pos=changed.index(before,cursor);spans.append((pos,before,after))
            changed=changed[:pos]+after+changed[pos+len(before):];cursor=pos+len(after)
    inverse=changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)]!=after:
            raise RuntimeError('Positional ordinary40 score-watch inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire completed saved-score supervisor restored byte-exact')
    return changed,dict(parent=prior,owner_sha256=_G40SW_OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),source_splices=len(spans),
        original_sampling_reaping_finalization_unchanged=True,whole_s=120,combined_rss_bytes=1_000_000_000,
        original40_flat_truth_hash_allowlist_only=True,current_native41_and_panel_objective_required=True)


if __name__=='__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert; explicit --run and reviewed current-source/actual result hashes required.')
    source,proof=derive_source()
    exec(compile(source,str(_G40SW_OWNER)+':current-ordinary40-score-watch','exec'),globals())
