"""TAG opus_engineer2 a62 make_idc_v1 (file written 2026-10-12T23:26:30Z; lead 22:5xZ list-D hunt, 23:2xZ 'keep it': the combined identity-cache cut, about
0.026% of the bill, byte-identical by construction): the content probe (probe_content_repeats_v1.py on s51k1tw_x 7100136) found the same op
recomputed on the SAME, never-written input objects:
  A  _ott_trace: norm2 = sum(F * F, axis=0) of the old tier's F (rebound at joins, never written in place: 'immutable base') on every
     layer: 0.017% (7 of 10 calls repeat);
  B  _ynt_trace: normB = sum(base * base, axis=0) of the newborns' common leg, already computed by _f4c_birth in the same layer: 0.003%;
  C  _np_update: cc32 = cov32 * cov32 of the layer's float32 covariance, already computed by _bd_update (C32 * C32, the same object): 0.006%.
Each repeat is served from a small identity cache (the owner object is HELD, so its identity stays its own; only the newest 'keep' entries per
tag stay, so memory stays bounded: one layer for B / C, three F's for A). The cached arrays are never held by a call's _E2Hold (which frees its
temporaries), the products' temporaries stay in the caller's hold as before. Same ops on the same operands: byte-identical predictions.
The functions are COPIED from the parent's LIVE definitions (the last top-level def of each name, found by ast) and edited by exact-once
replacements; the fragment records each original's sha. usage: make_idc_v1.py <parent.py> <out name>  -> s49/<out name>/candidate.py"""
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent, name = Path(sys.argv[1]).resolve(), sys.argv[2]
text = parent.read_text()
lines = text.split('\n')
tree = ast.parse(text)
live = {}
for node in tree.body:   # top level, in file order: the last def of a name is the live one
    if isinstance(node, ast.FunctionDef):
        live[node.name] = (node.lineno, node.end_lineno)
EDITS = {
    '_ott_trace': [("    norm2 = fnp.sum(F * F, axis=0)\n", "    norm2 = _idc_get('ott_n2', F, 3, lambda: fnp.sum(F * F, axis=0))   # opus-engineer2 idc A\n")],
    '_f4c_birth': [("_e2_H(fnp.sum(_e2_H(base * base), axis=0))", "_idc_get('colsq', base, 1, lambda: fnp.sum(_e2_H(base * base), axis=0))")],
    '_ynt_trace': [("    normB = fnp.sum(base * base, axis=0)\n", "    normB = _idc_get('colsq', base, 1, lambda: fnp.sum(base * base, axis=0))   # opus-engineer2 idc B\n")],
    '_bd_update': [("_e2_H(C32 * C32)", "_idc_get('cc32', C32, 1, lambda: C32 * C32)")],
    '_np_update': [("        cc32 = hold(cov32 * cov32)\n", "        cc32 = _idc_get('cc32', cov32, 1, lambda: cov32 * cov32)   # opus-engineer2 idc C\n")],
}
blocks, shas = [], []
for fn, edits in EDITS.items():
    a, b = live[fn]
    src = '\n'.join(lines[a - 1:b]) + '\n'
    shas.append('%s %s (lines %d-%d)' % (fn, hashlib.sha256(src.encode()).hexdigest()[:12], a, b))
    for old, new in edits:
        assert src.count(old) == 1, (fn, old, src.count(old))
        src = src.replace(old, new)
    blocks.append(src)
helper = '''# opus-engineer2 a62 IDENTITY CACHES idc v1 (make_idc_v1.py from %s, sha %s): the live definitions of %s, copied with the repeats
# served from _idc_get. Originals: %s
_IDC = {}
_IDC_STATS = dict(hits=0, misses=0)


def _idc_get(tag, owner, keep, make):
    entries = _IDC.setdefault(tag, [])
    for held, value in entries:
        if held is owner:
            _IDC_STATS['hits'] += 1
            return value
    value = make()
    entries.append((owner, value))
    if len(entries) > keep:
        del entries[0]
    _IDC_STATS['misses'] += 1
    return value


''' % (parent, hashlib.sha256(text.encode()).hexdigest()[:12], ', '.join(EDITS), '; '.join(shas))
frag = HERE / ('fragment_idc_%s.py' % name)
assert not frag.exists(), frag
frag.write_text(helper + '\n\n'.join(blocks))
compile(text + '\n' + frag.read_text(), str(frag), 'exec')
out = subprocess.run([sys.executable, '-B', str(HERE / 'make_s49_v1.py'), str(parent), name, str(frag)], capture_output=True, text=True)
assert out.returncode == 0, out.stderr[-500:]
cand = HERE / 's49' / name / 'candidate.py'
print('%s %s: idc A/B/C on %s; fragment %s; %s' % (name, hashlib.sha256(cand.read_bytes()).hexdigest()[:8], parent.parent.name, frag.name, '; '.join(shas)))
