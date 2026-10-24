#!/usr/bin/env python3
"""TAG fable-porter (PT-14): native OLD-TIER FRAME RANK as ONE literal. Count-asserted edits of a finished candidate (e.g. S11 acf67b9d):
(1) module constant `_TIER_RANK = R` placed with the lazy-rotation fragment's `_SLR_RANK = 384`, which becomes `_SLR_RANK = _TIER_RANK`
    (the lazy factor/map shape asserts); (2) in the LIVE (last) `_shared_old384_push` only: `rank = 384` -> `rank = _TIER_RANK` and the
    alive-row guard `< 384` -> `< _TIER_RANK`. The frame sketch (Omega = W[previous, :rank]), QR, factors and core follow the rank, as in
    the torch cards (s2card.make_push rank=, covhunter's s6bcard_r320). The rotation-envelope width literal 384.0 is left: it is a
    conservative overflow bound for any R <= 384 (hence R > 384 is refused). R = 384 must reproduce the input's predictions and bill.
usage: build_tier_rank.py <in.py> <out.py> <R>"""
import sys, hashlib
from pathlib import Path
src_path, out_path, R = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
assert 0 < R <= 384 and R % 32 == 0, 'R must be a positive multiple of 32 and <= 384'
assert not out_path.exists(), f'refusing to overwrite {out_path}'
src = src_path.read_text()
i = src.rfind('\ndef _shared_old384_push('); j = src.find('\ndef ', i + 5)
assert i > 0 and j > i
fn = src[i:j]
for old, new in (("    rank = 384\n", "    rank = _TIER_RANK\n"),
                 ("int(previous.shape[0]) < 384:\n", "int(previous.shape[0]) < _TIER_RANK:\n")):
    assert fn.count(old) == 1, ('live push anchor', old, fn.count(old))
    fn = fn.replace(old, new)
out = src[:i] + fn + src[j:]
old = "_SLR_RANK = 384\n"
assert out.count(old) == 1, ('_SLR_RANK anchor', out.count(old))
out = out.replace(old, f"_TIER_RANK = {R}   # PT-14 old-tier frame rank (one literal)\n_SLR_RANK = _TIER_RANK\n")
out_path.write_text(out)
print(dict(out=str(out_path), R=R, sha256=hashlib.sha256(out.encode()).hexdigest(), base=hashlib.sha256(src.encode()).hexdigest()[:16], edits=3))
