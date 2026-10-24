"""[opus-ms] builds: V-A+B 388cdd4d + editor_ms_v1 (the iffy line) and S4 452bda56 + editor_ms_v1 (the clean line)."""
import hashlib, importlib.util
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
s = importlib.util.spec_from_file_location('ems', D/'editor_ms_v1.py'); ed = importlib.util.module_from_spec(s); s.loader.exec_module(ed)
for name, path, sha in (('vab', R/'opus_passrel_v1/frontier_v1/cand_frontier_VAB_v1.py', '388cdd4d'), ('s4', R/'opus_passrel_v1/stack_v1/cand_s4.py', '452bda56')):
    t0 = path.read_text(); assert hashlib.sha256(t0.encode()).hexdigest().startswith(sha)
    t, counts = ed.apply(t0)
    a, b = t0.split('\n'), t.split('\n')
    ch = [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]
    o = D/f'cand_{name}_ms_v1.py'
    with o.open('x') as f: f.write(t)
    compile(t, str(o), 'exec')
    print(o.name, hashlib.sha256(t.encode()).hexdigest()[:8], 'counts', counts, 'changed lines', ch, 'lines', len(a), '->', len(b))
