"""Target-free algebra/autograd check before any fit is allowed."""
import hashlib
import json
from pathlib import Path
import resource
import time
from model import certificate, torch, np

HERE = Path(__file__).resolve().parent
started = time.monotonic()
result = certificate()
result.update(model_sha256=hashlib.sha256((HERE / 'model.py').read_bytes()).hexdigest(),
              certificate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              numpy_version=np.__version__, torch_version=torch.__version__, device='cpu',
              threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
              elapsed_s=time.monotonic() - started,
              max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE / 'algebra_certificate.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result), flush=True)
