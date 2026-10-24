"""Version the unfinished separate-frame study onto the S29 parameters."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
source = P2 / 'fable_covhunter_v1/r320_split2.py'
text = source.read_text()
changes = {
    "HERE = Path(__file__).resolve().parent; P2 = HERE.parent":
        "HERE = Path(__file__).resolve().parent; P2 = HERE.parent",
    "R3 = P2 / 'fable_physicist_v1/refit6b/s6b_loopR3_joint_s23'":
        "R3 = P2 / 'fable_physicist_v1/refit6b/s6b_loopR7s_joint_3ep'",
    "R3 / 'dials_epoch0.json'": "R3 / 'dials_epoch1.json'",
    "R3 / 'params_epoch0.pt'": "R3 / 'params_epoch1.pt'",
}
for old, new in changes.items():
    assert text.count(old) == 1, old
    text = text.replace(old, new)
target = HERE / 'split_screen_s29.py'
with target.open('x') as f:
    f.write(text)
manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                target=str(target), target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                changes=changes, claim='A screening surrogate, not native score evidence.')
(HERE / 'split_screen_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(target)
