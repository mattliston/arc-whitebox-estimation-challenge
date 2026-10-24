"""Original parameter-only exporter with explicit repaired-run provenance."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
def fixed(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

repair_path = HERE / 'runtime_repair_v2.json'
admission = json.loads((HERE / 'runtime_v2_root_admission.json').read_text())
assert admission['root_admitted'] is True and admission['repair'] == fixed(repair_path)
assert admission['original_protocol'] == fixed(HERE / 'protocol.json')
repair = json.loads(repair_path.read_text())
for record in repair['pins'] + repair['preserved_failure']:
    assert fixed(record['path']) == record, record['path']
source = HERE / 'export.py'
assert fixed(source)['sha256'] == '80f03540f522c758e1cd747c1f5328b9cf0369ed94a0e73fde84dbf2f1210e03'
text = source.read_text()
changes = [
    ("HERE / 'training/completed.json'", "HERE / 'training_runtime_v2/completed.json'", 1),
    ("assert done['trainer'] == pin(HERE / 'train.py') and done['protocol'] == pin(protocol_path)",
     "assert done['trainer'] == pin(HERE / 'train_runtime_v2.py') and done['protocol'] == pin(protocol_path)\n"
     "assert done['original_trainer'] == pin(HERE / 'train.py')\n"
     "assert done['runtime_repair'] == pin(HERE / 'runtime_repair_v2.json')", 1),
    ("HERE / 'training_guard.json'", "HERE / 'training_runtime_v2_guard.json'", 2),
    ("str(HERE / 'train.py')", "str(HERE / 'train_runtime_v2.py')", 1),
    ('exporter=pin(__file__), protocol=pin(protocol_path),',
     "exporter=pin(__file__), original_exporter=pin(HERE / 'export.py'), "
     "trainer=done['trainer'], original_trainer=done['original_trainer'], "
     "runtime_repair=pin(HERE / 'runtime_repair_v2.json'), "
     "runtime_admission=pin(HERE / 'runtime_v2_root_admission.json'), protocol=pin(protocol_path),", 1),
]
for old, new, count in changes:
    assert text.count(old) == count, old
    text = text.replace(old, new)
exec(compile(text, str(source) + ':runtime_v2', 'exec'), globals())
