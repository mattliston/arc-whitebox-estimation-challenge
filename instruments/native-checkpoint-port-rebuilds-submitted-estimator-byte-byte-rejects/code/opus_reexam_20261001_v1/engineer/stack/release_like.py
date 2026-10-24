"""TAG opus_engineer (2026-10-11): a release-lane variant built IN THIS LANE, byte for byte as release/build_variant.py builds it (parent source
+ the named release fragments appended at the end, same header lines), so that censuses and A/B sessions run on the configuration the
release fires (e.g. Rprices_zero changes the planners' depths and so the tree keys). Reads release/variants/fragments/<name>.py (read-only);
writes stack/rl/<name>/candidate.py + build.json; refuses to overwrite. usage: release_like.py <parent.py> <name> <fragment> [<fragment> ...]"""
import hashlib
import json
from pathlib import Path
import sys

E = Path(__file__).resolve().parents[1]
FR = E.parent/'release'/'variants'/'fragments'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent = Path(sys.argv[1]).resolve()
name = sys.argv[2]
out = E/'stack'/'rl'/name
out.mkdir(parents=True)
text = parent.read_text()
if not text.endswith('\n'):
    text += '\n'
used = []
for fragment in sys.argv[3:]:
    path = FR/f'{fragment}.py'
    body = path.read_text()
    text += f"\n\n# BEGIN OPUS-RELEASE FRAGMENT {fragment} (sha256 {sha(path)[:12]})\n{body.rstrip()}\n# END OPUS-RELEASE FRAGMENT {fragment}\n"
    used.append(dict(name=fragment, path=str(path), sha256=sha(path)))
compile(text, str(out/'candidate.py'), 'exec')
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent), fragments=used,
                                              candidate=str(out/'candidate.py'), candidate_sha256=sha(out/'candidate.py')), indent=1) + '\n')
print(out/'candidate.py', sha(out/'candidate.py')[:12])
