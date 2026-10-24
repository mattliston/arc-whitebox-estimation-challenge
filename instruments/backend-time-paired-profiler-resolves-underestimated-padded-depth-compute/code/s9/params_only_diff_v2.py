#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-PD v2: unary minus before a literal number is part of the value): is <b.py> the same program as <a.py> up to literal VALUES? Tokenizes both files, masks every NUMBER
token and every STRING token longer than 40 chars (parameter blobs), and compares the token streams. Reports the first code-token
differences (if any) and counts of changed numbers / long strings. usage: params_only_diff.py <a.py> <b.py>"""
import sys, tokenize, io
def toks(path):
    out = []
    with open(path, 'rb') as f:
        for t in tokenize.tokenize(f.readline):
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            out.append(t)
    merged = []   # v2: a unary minus directly before a NUMBER (after an opener, comma, '=' or ':') is part of the literal value
    for i, t in enumerate(out):
        if (t.type == tokenize.OP and t.string == '-' and i + 1 < len(out) and out[i + 1].type == tokenize.NUMBER
                and i > 0 and out[i - 1].type == tokenize.OP and out[i - 1].string in ('(', '[', '{', ',', '=', ':')):
            continue
        merged.append(t)
    return merged
a, b = toks(sys.argv[1]), toks(sys.argv[2])
def key(t):
    if t.type == tokenize.NUMBER: return ('NUM',)
    if t.type == tokenize.STRING and len(t.string) > 40: return ('LONGSTR',)
    return (t.type, t.string)
ka, kb = [key(t) for t in a], [key(t) for t in b]
if ka == kb:
    nnum = sum(1 for x, y in zip(a, b) if x.type == tokenize.NUMBER and x.string != y.string)
    nstr = sum(1 for x, y in zip(a, b) if x.type == tokenize.STRING and len(x.string) > 40 and x.string != y.string)
    lines = sorted({x.start[0] for x, y in zip(a, b) if x.string != y.string})
    print(f"PARAMS-ONLY: identical code tokens ({len(a)}); changed numbers {nnum}, changed long strings {nstr}; changed lines in a: {len(lines)} (first {lines[:8]})")
else:
    import difflib
    sm = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
    diffs = [op for op in sm.get_opcodes() if op[0] != 'equal']
    print(f"CODE DIFFERS: {len(diffs)} differing token runs; first ones:")
    for tag, i1, i2, j1, j2 in diffs[:8]:
        la = a[i1].start[0] if i1 < len(a) else -1; lb = b[j1].start[0] if j1 < len(b) else -1
        print(f"  {tag}: a line {la} {' '.join(t.string for t in a[i1:min(i2, i1+12)])[:150]!r} -> b line {lb} {' '.join(t.string for t in b[j1:min(j2, j1+12)])[:150]!r}")
