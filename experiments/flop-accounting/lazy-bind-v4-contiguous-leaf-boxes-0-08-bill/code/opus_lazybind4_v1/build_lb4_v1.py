"""[opus-lazybind] v4 builds: S3 6921feaa (release stack_v1/cand_s3.py) + fragment_lazybind_v4.py.txt appended LAST."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_passrel_v1/stack_v1/cand_s3.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('6921feaa')
frag = (D/'fragment_lazybind_v4.py.txt').read_text()
o = D/'cand_s3_lb4_v1.py'
with o.open('x') as f: f.write(src + frag)
compile(o.read_text(), str(o), 'exec'); print(o.name, hashlib.sha256(o.read_bytes()).hexdigest()[:8], 'frag', hashlib.sha256(frag.encode()).hexdigest()[:8])
