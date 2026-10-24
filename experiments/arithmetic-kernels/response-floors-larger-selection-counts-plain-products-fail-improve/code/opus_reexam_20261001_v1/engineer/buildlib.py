"""TAG opus_engineer build helper: derive a candidate from a parent by count-asserted text edits and/or an appended fragment.

build(parent, out, edits=[(old, new, expected_count)], append='...') writes out (never overwrites), plus out + '.build.json'
with both sha256 values, so every attempt directory holds the exact derivation from RRPI."""
import hashlib
import json
from pathlib import Path

P2 = Path(__file__).resolve().parents[2]
H = P2/'codex_resume_20260926_v2'
RRPI = H/'hill_climb_pair_rr_inverse_v1/candidate.py'
RRPI_SHA = 'a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(parent, out, edits=(), append='', expect_parent_sha=None):
    parent, out = Path(parent), Path(out)
    assert not out.exists(), out
    text = parent.read_text()
    psha = hashlib.sha256(text.encode()).hexdigest()
    if expect_parent_sha:
        assert psha == expect_parent_sha, psha
    for old, new, count in edits:
        assert text.count(old) == count, (old[:80], text.count(old), count)
        text = text.replace(old, new)
    if append:
        text = text.rstrip('\n') + '\n\n\n' + append.strip('\n') + '\n'
    compile(text, str(out), 'exec')
    out.write_text(text)
    record = dict(parent=str(parent), parent_sha256=psha, out=str(out), sha256=sha(out), bytes=out.stat().st_size,
                  edits=[dict(old=o, new=n, count=c) for o, n, c in edits], append_sha256=hashlib.sha256(append.encode()).hexdigest() if append else None)
    Path(str(out) + '.build.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(dict(out=str(out), sha256=record['sha256'], bytes=record['bytes'])))
    return out
