"""Filename/schema-only inverse of the preserved independent14 score watcher."""
import hashlib
import importlib.util
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_V1_WATCH = _HERE/'watch_score_panel.py'
_V1_WATCH_SHA = '2c3cdbcd317aad378a2eaa5c36b2420247d74abb70eeff6d401b181a037de931'
# The inherited resource body's _OWNER binding remains its original ce125 owner.
_OWNER = _HERE/'watch_score_probe.py'
_OWNER_SHA = 'ce125eeb2bf774f9647b925018ba3ea1bf4e7b0a4e18277d6fab2d1c09d7d6a5'


def derive_source():
    if _V1_WATCH.is_symlink() or hashlib.sha256(_V1_WATCH.read_bytes()).hexdigest() != _V1_WATCH_SHA:
        raise RuntimeError('Immutable original failed panel-score watcher')
    spec = importlib.util.spec_from_file_location('shared_panel_watch_v1_source_only',_V1_WATCH)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    original,prior = parent.derive_source()
    if parent._OWNER != _OWNER or parent._OWNER_SHA != _OWNER_SHA:
        raise RuntimeError('Exact original live resource-loop owner binding')
    edits = (
        ("HERE/'score_panel.py'","HERE/'score_panel_v2.py'",3),
        ("HERE/'watch_score_panel.py'","HERE/'watch_score_panel_v2.py'",1),
        ("HERE/'SCORE_PANEL_PROTOCOL.md'","HERE/'SCORE_PANEL_V2_PROTOCOL.md'",1),
        ("HERE/'SCORE_PANEL_WATCH_SPEC.md'","HERE/'SCORE_PANEL_V2_WATCH_SPEC.md'",1),
        ("'current_comp_shared_old384_score_panel_watch_v1'","'current_comp_shared_old384_score_panel_watch_v2'",1),
    )
    changed = original
    for old,new,count in edits:
        if changed.count(old) != count or new in changed:
            raise RuntimeError('Exact v2 watcher source boundary: '+old)
        changed = changed.replace(old,new)
    inverse = changed
    for old,new,count in reversed(edits):
        if inverse.count(new) != count:
            raise RuntimeError('Exact v2 watcher inverse boundary: '+new)
        inverse = inverse.replace(new,old)
    if inverse != original:
        raise RuntimeError('Whole original panel-score resource body must be recovered')
    return changed,dict(parent=prior,owner_sha256=_V1_WATCH_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        versioned_paths_and_schema_only=True,whole_s=120,combined_rss_bytes=2_000_000_000,
        original_sampling_reaping_finalization_and_truth_byte_allowlist_unchanged=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert; explicit --run and reviewed v2 hashes are required.')
    source,proof = derive_source()
    exec(compile(source,str(_V1_WATCH)+':panel-score-watch-v2','exec'),globals())
