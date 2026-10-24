"""TAG opus_engineer a62 (file written 2026-10-12 16:44Z; lead 16:5xZ: native check of the birth-floor arms of the physicist's T129): the BIRTH floor by
layer. Native births are chosen by select(min(g, 1 - g), gate_floor) in the core loop (the only active birth selection; the response and alive
selections keep gate_floor: T129's arms move births only, its response / alive counts are identical across arms). The births line is edited
by TEXT at its last occurrence to select(_e2_gate_min, _BFLOOR.get(l, gate_floor)), plus a block _BFLOOR = {} (= the parent bit for bit);
an optional values fragment sets {layer: floor}.
usage: make_floor_v1.py <parent.py> <out name> [<fragment_bfloor_<name>.py>]   -> a62_modules/s49/<out name>/candidate.py + build.json"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent, name = Path(sys.argv[1]).resolve(), sys.argv[2]
values = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
out = HERE/'s49'/name
out.mkdir(parents=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
OLD = 'selected, births = select(_e2_gate_min, gate_floor)'
NEW = 'selected, births = select(_e2_gate_min, _BFLOOR.get(l, gate_floor))'
lines = parent.read_text().split('\n')
hits = [i for i, s in enumerate(lines) if s.strip().startswith(OLD)]
assert hits, 'births selection not found'
i = hits[-1]
lines[i] = lines[i].replace(OLD, NEW) + '   # opus-engineer a62 birth floor by layer'
print('births line: %d occurrence(s), edited the last at line %d' % (len(hits), i + 1))
text = '\n'.join(lines).rstrip('\n') + '\n\n# BEGIN OPUS-ENGINEER a62 BIRTH FLOOR v1 (make_floor_v1.py)\n_BFLOOR = {}\n# END OPUS-ENGINEER a62 BIRTH FLOOR v1\n'
if values is not None:
    body = values.read_text()
    text += '\n# BEGIN OPUS-ENGINEER A62 FRAGMENT %s (sha256 %s)\n%s\n# END OPUS-ENGINEER A62 FRAGMENT %s\n' % (values.stem, sha(body.encode())[:12], body.strip('\n'), values.stem)
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent.read_bytes()), edited_line=i + 1, values=None if values is None else str(values),
                                               candidate=str(out/'candidate.py'), candidate_sha256=sha(text.encode())), indent=1) + '\n')
print(name, sha(text.encode())[:8])
