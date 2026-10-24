"""TAG scout-native check_stack_vs_reference_v2 (2026-10-13) = v1 + COPIES: a fragment function under a NEW name (e.g. NPX's _npx_left, a
copy of _np_left) is matched to the reference prefix's most similar live function (Jaccard of stripped lines >= 0.5), and that source
function's live text is compared between the two prefixes (a change there means the copy would drop a parent change). v1's text follows.
TAG scout-native check_stack_vs_reference_v1 (2026-10-13, lead 07:14Z: the S53 stack carries SN3, rows2, npx1, Astra's block, idc1, k4lf,
HZT, ASZ): every carried fragment's context on the new stack vs the context it was certified in on the reference (the fired twin). For each
fragment marker found in both files (idc1 pairs by role: its versions differ per parent), the prefix of each file before the marker; for
every name the fragment DEFINES that the prefix already defines (an override: a copied body would revert a parent change), the live (last
top-level) definition in the two prefixes, SAME or DIFF; then the names the fragment only READS whose live text differs (parent behaviour the
fragment calls: dials / round changes, information only).
usage: check_stack_vs_reference_v1.py <stack.py> <reference.py>"""
import ast
import re
import sys


def load(path):
    lines = open(path).read().split('\n')
    marks = {}
    for i, line in enumerate(lines):
        m = re.match(r'# BEGIN OPUS-ENGINEER A62 FRAGMENT (\S+) ', line)
        if m:
            marks[re.sub(r'^fragment_idc_idc1.*', 'fragment_idc_idc1', m.group(1))] = i
    return lines, marks


def live(lines):
    out = {}
    for n in ast.parse('\n'.join(lines)).body:
        if isinstance(n, ast.FunctionDef):
            names = [n.name]
        elif isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
        else:
            continue
        for x in names:
            out[x] = '\n'.join(lines[n.lineno - 1:n.end_lineno])
    return out


def fragment_functions(lines, start):
    end = next(i for i in range(start, len(lines)) if lines[i].startswith('# END OPUS-ENGINEER A62 FRAGMENT'))
    body = lines[start:end]
    return {n.name: '\n'.join(body[n.lineno - 1:n.end_lineno]) for n in ast.parse('\n'.join(body)).body if isinstance(n, ast.FunctionDef)}


def alldefs(lines):   # every top-level def in file order: name -> [text of the 1st, 2nd, ... definition] (earlier ones may be captured)
    out = {}
    for n in ast.parse('\n'.join(lines)).body:
        if isinstance(n, ast.FunctionDef):
            out.setdefault(n.name, []).append('\n'.join(lines[n.lineno - 1:n.end_lineno]))
    return out


def copied_from(text, prefix_defs):
    a = {s.strip() for s in text.split('\n')[1:] if s.strip() and not s.strip().startswith('#')}
    best, score = None, 0.0
    for name, texts in prefix_defs.items():
        for k, other in enumerate(texts):
            b = {s.strip() for s in other.split('\n')[1:] if s.strip() and not s.strip().startswith('#')}
            j = len(a & b) / max(1, len(a | b))
            if j > score:
                best, score = (name, k), j
    return best, score


def fragment_names(lines, start):
    end = next(i for i in range(start, len(lines)) if lines[i].startswith('# END OPUS-ENGINEER A62 FRAGMENT'))
    tree = ast.parse('\n'.join(lines[start:end]))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    defined |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    read = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return defined, read - defined


new, ref = load(sys.argv[1]), load(sys.argv[2])
for name in sorted(set(new[1]) & set(ref[1]), key=lambda k: new[1][k]):
    a, b = new[1][name], ref[1][name]
    pa, pb = live(new[0][:a]), live(ref[0][:b])
    defined, read = fragment_names(new[0], a)
    over = sorted(x for x in defined if x in pa or x in pb)
    bad = [x for x in over if pa.get(x) != pb.get(x)]
    info = sorted(x for x in read if x in pa and pa.get(x) != pb.get(x))
    copies = []
    da, db = alldefs(new[0][:a]), alldefs(ref[0][:b])
    for fn, text in fragment_functions(new[0], a).items():
        if fn in pa or fn in pb:
            continue
        src, score = copied_from(text, db)
        if src and score >= 0.5:
            sn, k = src
            same = len(da.get(sn, [])) == len(db[sn]) and da[sn][k] == db[sn][k]
            copies.append('%s<-%s#%d/%d %.2f %s' % (fn, sn, k + 1, len(db[sn]), score, 'SAME' if same else 'DIFF'))
    print('%-28s overrides %d: %s | copies: %s | reads that differ: %s' % (name, len(over), 'DIFF ' + ' '.join(bad) if bad else 'all SAME', ', '.join(copies) or 'none', ' '.join(info) or 'none'))
print('only in the stack:', sorted(set(new[1]) - set(ref[1])), '| only in the reference:', sorted(set(ref[1]) - set(new[1])))
