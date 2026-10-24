"""Common-cubic native zero3/terminal5; unchanged original resource guard."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_receiver_directions_v1/watch_native.py'
_parent_sha = '5637a5e24411aa74a989206709c9c00c695f279b4941792672522fa68de13947'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Frozen native supervisor source owner changed')
_original = _parent.read_text()
_before, _after = 'receiver_R_initialization_owned', 'common_Q_initialization_owned'
if _original.count(_before) != 1 or _after in _original:
    raise RuntimeError('Unique common-cubic startup ownership assertion')
_derived = _original.replace(_before, _after)
if _derived.replace(_after, _before) != _original:
    raise RuntimeError('Whole native supervisor source inverse failed')
exec(compile(_derived, str(_parent)+':common-cubic-native-watch', 'exec'),
     {'__name__': __name__, '__file__': __file__})
