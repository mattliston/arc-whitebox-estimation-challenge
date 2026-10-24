"""v2: byte-exact source replay patch (base -> target), SOURCE_REPLAY_PATCH_v1 schema.
Hunks are built in order against the progressively patched text, so each hunk's 'before' context is unique
in exactly the text that sequential replay will see (v1 checked uniqueness against the base only, which fails
when appended fragments duplicate base code). No setup, prediction or live activation."""
import argparse, difflib, hashlib, json
from pathlib import Path

sha = lambda b: hashlib.sha256(b).hexdigest()


def build(base, target):
    cur, b = base.splitlines(keepends=True), target.splitlines(keepends=True)
    hunks, offset = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, base.splitlines(keepends=True), b, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        s, e = i1 + offset, i2 + offset
        text = ''.join(cur)
        lo, hi = s, e
        while True:
            before = ''.join(cur[lo:hi])
            if before and text.count(before) == 1:
                break
            lo, hi = max(lo - 1, 0), min(hi + 1, len(cur))
        new = cur[lo:s] + b[j1:j2] + cur[e:hi]
        hunks.append(dict(before=before, after=''.join(new)))
        cur = cur[:lo] + new + cur[hi:]
        offset += (j2 - j1) - (i2 - i1)
    return hunks


def replay(text, hunks):
    for h in hunks:
        assert text.count(h['before']) == 1, 'Source context drift; do not silently port'
        text = text.replace(h['before'], h['after'], 1)
    return text


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--target', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    x = ap.parse_args()
    base, target = x.base.read_bytes(), x.target.read_bytes()
    hunks = build(base.decode(), target.decode())
    assert replay(base.decode(), hunks).encode() == target
    patch = dict(base=str(x.base), base_sha256=sha(base), target=str(x.target), target_sha256=sha(target), hunks=hunks)
    with x.output.open('x') as f:
        json.dump(patch, f, indent=1)
        f.write('\n')
    print(json.dumps(dict(output=str(x.output), output_sha256=sha(x.output.read_bytes()), base_sha256=sha(base), target_sha256=sha(target), hunks=len(hunks), replay_exact=True)))
