"""Offline environment/source verification. Run once in each Linux SDK environment."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import platform
from common_linux_v1 import check_sdk, load_plan, sha, utc, write_new

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True); p.add_argument('--plan', type=Path, required=True)
p.add_argument('--role', choices=['client', 'server'], required=True); p.add_argument('--out', type=Path, required=True)
a = p.parse_args(); plan = load_plan(a.root, a.plan)
record = check_sdk(a.root, plan, a.role)
record.update(utc=utc(), machine=platform.machine(), release=platform.release(), plan_sha256=sha(a.plan))
if a.role == 'client':
    assert importlib.util.find_spec('numpy') is None
    record['numpy_absent'] = True
else:
    import numpy as np
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        np.show_config()
    record['numpy_config'] = text.getvalue()
    libraries = sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                           if any(x in line.lower() for x in ('openblas', 'mkl_rt', 'libblas', 'libblis'))))
    record['loaded_blas_libraries'] = [dict(path=x, sha256=sha(x)) for x in libraries]
    assert libraries, 'record actual loaded BLAS before calibration'
write_new(a.out, record)
print(json.dumps(dict(role=a.role, out=str(a.out), sdk_source_hashes_verified=True)))
