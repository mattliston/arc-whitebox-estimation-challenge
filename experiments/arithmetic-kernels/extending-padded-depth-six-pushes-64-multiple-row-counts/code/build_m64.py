#!/usr/bin/env python3
"""TAG fable-engineer (E-51 --m64, 2026-09-24): six-level pushes for ANY row count m that is a multiple of 64 (m <= 1024), not only
m == 1024. Post-stage: apply it LAST (after build_colpack and build_deadcut). Every product with m == 1024 takes the unchanged code path
(bit-exact); pushes with 64 <= m < 1024, m % 64 == 0 (deadcut's kept rows with --round=64) may now take six levels:
  - _rule_depth: for those m the v3 rule's own pricing (bill ratio + MU * fitted seconds, padded six priced with its pad copies) chooses
    among four, five and six levels, exactly as it does at m == 1024;
  - _mm: the zero-padded six-level path (_push6_padded) accepts m % 64 == 0 (it pads k and n only);
  - _packed_product_4: m % 64 == 0 rows run the batched six-level layout (packets of 4096 columns; with --colpack the left transform is
    shared across the packets, as at m == 1024).
usage: build_m64.py <in.py> <out.py>"""
import sys, hashlib
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists():
    sys.exit(f"refusing to overwrite {dst}")
s = src.read_text()
assert "E-51 --m64" not in s, "input already has --m64"
for name in ("_mm", "_rule_depth", "_packed_product_4"):
    assert s.count("\ndef " + name + "(") >= 1, name
def live_def(text, name):
    i = text.rfind("\ndef " + name + "(")
    lines = text[i + 1:].split("\n"); out = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() == "" or ln.startswith((" ", "\t")):
            out.append(ln)
        else:
            break
    while out and out[-1].strip() == "":
        out.pop()
    return "\n".join(out)
mm = live_def(s, "_mm")
old = "    if n > 1024 and rd == 6 and m == 1024 and (k % 64 or n % 64):\n"
assert mm.count(old) == 1, "live _mm padded-six condition"
mm = mm.replace(old, "    if n > 1024 and rd == 6 and m % 64 == 0 and (k % 64 or n % 64):   # E-51 --m64: any m % 64 == 0 (was m == 1024)\n")
frag = '''

# BEGIN FABLE-ENGINEER E-51 --m64 (2026-09-24): six-level pushes for m % 64 == 0 rows (see build_m64.py). m == 1024: unchanged paths.
_M64_RULE = _rule_depth
_M64_CACHE = {}


def _rule_depth(kind, m, k, n, maximum=6):
    d = _M64_RULE(kind, m, k, n, maximum)
    if not (kind == 'push' and d in (4, 5) and 64 <= m < 1024 and m % 64 == 0 and k <= 1024 and k % 32 == 0 and n % 32 == 0
            and maximum >= 6 and (kind, m, k, n) not in _DR_FORCE):
        return d
    key = (m, k, n)
    found = _M64_CACHE.get(key)
    if found is None:
        options = (4, 5, 6) if (k % 64 == 0 and n % 64 == 0) else (4, 5, 7)
        def value(L):
            if L != 7:
                return _dr_value('push', m, k, n, L, _DR_MU)
            kp, npad = (k + 63) // 64 * 64, (n + 63) // 64 * 64
            return _dr_value('push', m, kp, npad, 6, _DR_MU) + m * k + k * n + _DR_MU * (m * kp + kp * npad) / 2e9
        choice = min(options, key=value)
        found = _M64_CACHE[key] = 6 if choice == 7 else choice
    return found


''' + mm + '''


_M64_PP4 = _packed_product_4


def _packed_product_4(a, b, out, arena_views):
    m = int(a.shape[0]) if len(a.shape) == 2 else 0
    if (m != 1024 and 64 <= m < 1024 and m % 64 == 0
            and len(b.shape) == len(out.shape) == 2 and 0 < int(a.shape[1]) <= 1024
            and 1024 < int(b.shape[1]) <= 8192 and int(a.shape[1]) % 64 == int(b.shape[1]) % 64 == 0
            and int(b.shape[0]) == int(a.shape[1]) and tuple(out.shape) == (m, int(b.shape[1]))
            and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and _SCRATCH is not None and arena_views is _SCRATCH[1] and _SCRATCH_PUSH6 is not None):
        n6 = int(b.shape[1])
        if n6 > 4096:
            if '_cp_d6_shared' in globals():
                return _cp_d6_shared(a, b, out)
            _packed_product_cached(a, b[:, :4096], out[:, :4096], _SCRATCH_PUSH6, 6)
            _packed_product_cached(a, b[:, 4096:], out[:, 4096:], _SCRATCH_PUSH6, 6)
            return out
        return _packed_product_cached(a, b, out, _SCRATCH_PUSH6, 6)
    return _M64_PP4(a, b, out, arena_views)
# END FABLE-ENGINEER E-51 --m64
'''
s = s.rstrip("\n") + "\n" + frag
compile(s, str(dst), "exec")
dst.write_text(s)
print(f"{hashlib.sha256(s.encode()).hexdigest()} {dst.name}  (input {hashlib.sha256(src.read_bytes()).hexdigest()[:16]})")
