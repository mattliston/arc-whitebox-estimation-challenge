"""TAG opus-release: build a variant candidate = parent source + named fragments appended at the end (the estimator is a patch stack:
the LAST definition of a name wins, and module-level flags are read at call time, so a fragment that re-assigns a flag or re-defines
a function changes the behaviour without touching the parent's text). Writes variants/<name>/candidate.py and build.json; refuses
to overwrite. usage: build_variant.py <parent.py> <name> <fragment name> [<fragment name> ...]"""
import json
from pathlib import Path
import sys

from common import R, sha

parent = Path(sys.argv[1]).resolve()
name = sys.argv[2]
fragments = sys.argv[3:]
out = R/'variants'/name
out.mkdir()
text = parent.read_text()
if not text.endswith('\n'):
    text += '\n'
used = []
for fragment in fragments:
    path = R/'variants'/'fragments'/f'{fragment}.py'
    body = path.read_text()
    text += f"\n\n# BEGIN OPUS-RELEASE FRAGMENT {fragment} (sha256 {sha(path)[:12]})\n{body.rstrip()}\n# END OPUS-RELEASE FRAGMENT {fragment}\n"
    used.append(dict(name=fragment, path=str(path), sha256=sha(path)))
compile(text, str(out/'candidate.py'), 'exec')
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent), fragments=used,
                                              candidate=str(out/'candidate.py'), candidate_sha256=sha(out/'candidate.py')), indent=1) + '\n')
print(out/'candidate.py', sha(out/'candidate.py')[:12])
