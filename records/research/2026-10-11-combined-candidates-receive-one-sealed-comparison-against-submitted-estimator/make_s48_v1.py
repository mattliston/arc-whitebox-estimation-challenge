"""TAG opus_engineer a62 S48 builder (lead 16:5xZ GO: the combined file as separable fragments): the fired S47e release file
(stack/rl/s47rsd3x_safe/candidate.py = the S48 base s46h7rsd3x + the trainer's Nd values + the release fragments l3_bind128,
lvfs_limit_off, prices_zero) with this lane's a62 fragments appended in the given order, each between BEGIN / END lines carrying its
sha, the way release/build_variant.py appends release fragments. Any subset can be taken; the order does not matter between them.
An argument NAME=VALUE (e.g. _RC_COLZ=None) appends that one assignment after the fragments (certificates: neutral values).
usage: make_s48_v1.py <out name> <fragment or NAME=VALUE> [...]   -> a62_modules/s48/<out name>/candidate.py + build.json (no overwrite)"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent/'stack/rl/s47rsd3x_safe/candidate.py'
sha = lambda b: hashlib.sha256(b).hexdigest()
name, items = sys.argv[1], sys.argv[2:]
out = HERE/'s48'/name
out.mkdir(parents=True)
text = SRC.read_text()
assert sha(text.encode()).startswith('01103249')
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
