"""TAG opus_engineer a62 S49 builder = make_s48_v1.py with the PARENT as the first argument (a release-like file: a hand-over built by
stack/s48_build.sh on base s46h7rsd3x, then stack/release_like.py with the release fragments), for RAD-S48 and later points.
make_s48_v1 header: the fired S47e release file
(stack/rl/s47rsd3x_safe/candidate.py = the S48 base s46h7rsd3x + the trainer's Nd values + the release fragments l3_bind128,
lvfs_limit_off, prices_zero) with this lane's a62 fragments appended in the given order, each between BEGIN / END lines carrying its
sha, the way release/build_variant.py appends release fragments. Any subset can be taken; the order does not matter between them.
An argument NAME=VALUE (e.g. _RC_COLZ=None) appends that one assignment after the fragments (certificates: neutral values).
usage: make_s49_v1.py <parent.py> <out name> <fragment or NAME=VALUE> [...]   -> a62_modules/s49/<out name>/candidate.py + build.json"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
SRC, name, items = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3:]
out = HERE/'s49'/name
out.mkdir(parents=True)
text = SRC.read_text()
used = []
for item in items:
    if '=' in item and not item.endswith('.py'):
        text += '\n%s   # opus-engineer a62 make_s48_v1: neutral / override value\n' % item
        used.append(dict(assign=item))
        continue
    path = HERE/item
    body = path.read_text()
    text += '\n\n# BEGIN OPUS-ENGINEER A62 FRAGMENT %s (sha256 %s)\n%s\n# END OPUS-ENGINEER A62 FRAGMENT %s\n' % (
        path.stem, sha(body.encode())[:12], body.strip('\n'), path.stem)
    used.append(dict(name=path.stem, path=str(path), sha256=sha(body.encode())))
compile(text, str(out/'candidate.py'), 'exec')
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(SRC), parent_sha256=sha(SRC.read_bytes()), items=used,
                                              candidate=str(out/'candidate.py'), candidate_sha256=sha(text.encode())), indent=1) + '\n')
print(out/'candidate.py', sha(text.encode())[:8])
