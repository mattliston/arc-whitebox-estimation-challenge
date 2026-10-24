from pathlib import Path
import hashlib
source = Path(__file__).with_name('torch_probe.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'c257c57e36ea8e52704b400f1cb6201e87b9802b61db24b3f236a2cc4b989362'
s = source.read_text()
changes = [
 ("tensor = torch.from_numpy(value.copy())", "tensor = torch.from_numpy(np.array(value, copy=True))"),
 ("torch.from_numpy(directions[key].astype(np.float32))", "torch.from_numpy(np.asarray(directions[key], dtype=np.float32))"),
 ("out = HERE / ('torch_rank' + str(rank))", "out = HERE / ('torch_rank' + str(rank) + '_runtime_v2')"),
]
for old,new in changes:
    assert s.count(old) == 1, old
    s = s.replace(old,new)
exec(compile(s, str(source) + ':runtime_v2', 'exec'), globals())
