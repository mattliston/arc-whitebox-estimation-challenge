"""Small shared helpers. No SDK import, estimator execution, or network access."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')}


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def path_under(root, rel):
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('expected bundle-relative path: ' + str(rel))
    root = Path(root).resolve()
    result = (root / p).resolve()
    if not result.is_relative_to(root):
        raise ValueError('path escapes bundle: ' + str(rel))
    return result


def verified(root, record):
    p = path_under(root, record['path'])
    if 'bytes' in record and p.stat().st_size != record['bytes']:
        raise ValueError('size mismatch: ' + str(p))
    if sha(p) != record['sha256']:
        raise ValueError('SHA256 mismatch: ' + str(p))
    return p


def write_new(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as f:
        f.write(payload)


def no_governor(source):
    last = ast.parse(source).body[-1]
    assert isinstance(last, ast.Assign) and len(last.targets) == 1
    assert isinstance(last.targets[0], ast.Name) and last.targets[0].id == '_FAST_RESCUE_FRAC'
    assert isinstance(last.value, ast.Dict) and not last.value.keys, 'final empty rescue map required'


def candidate_path(root, record, out):
    p = verified(root, record)
    source = p.read_bytes()
    if record.get('append_tail'):
        tail = verified(root, record['append_tail']).read_bytes()
        source += b'\n' + tail
        p = Path(out) / 'candidate_frozen.py'
        with p.open('xb') as f:
            f.write(source)
    no_governor(source.decode())
    assert hashlib.sha256(source).hexdigest() == record['effective_sha256']
    return p


def load_plan(root, path):
    plan_path = Path(path).resolve()
    plan = json.loads(plan_path.read_text())
    verified(root, plan['transfer_manifest'])
    return plan


def check_sdk(root, plan, role):
    if sys.platform != 'linux' or sys.byteorder != 'little':
        raise RuntimeError('Linux little-endian environment required')
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('CPython3.10 required; actual ' + sys.version)
    if os.environ.get('FLOPSCOPE_WEIGHTS_FILE') or os.environ.get('FLOPSCOPE_DISABLE_WEIGHTS', '').lower() in ('1', 'true', 'yes', 'on'):
        raise RuntimeError('official packaged tariff required')
    versions = {name: importlib.metadata.version(name) for name in plan['packages'][role]['installed']}
    assert versions == plan['packages'][role]['installed'], versions
    import flopscope
    assert flopscope.__version__.split('+')[0] == '0.12.3', flopscope.__version__
    site = Path(flopscope.__file__).resolve().parent.parent
    for rec in plan['sdk_sources'][role]:
        p = path_under(site, rec['path'])
        assert sha(p) == rec['sha256'], 'SDK source/tariff mismatch: ' + str(p)
    return dict(python=sys.version, python_patch_matches_reference=sys.version.split()[0] == plan['packages'][role]['python'],
                packages=versions, flopscope=flopscope.__version__, sdk_source_hashes_verified=True,
                platform=sys.platform, byteorder=sys.byteorder)


def proc_stat(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    suffix = text[text.rfind(')') + 2:].split()
    ticks = os.sysconf('SC_CLK_TCK')
    return dict(pid=int(pid), state=suffix[0], ppid=int(suffix[1]), pgid=int(suffix[2]),
                start_ticks=int(suffix[19]), user_s=int(suffix[11]) / ticks,
                system_s=int(suffix[12]) / ticks, rss_bytes=int(suffix[21]) * os.sysconf('SC_PAGE_SIZE'),
                minor_faults=int(suffix[7]), major_faults=int(suffix[9]))


def own_usage():
    import resource
    u = resource.getrusage(resource.RUSAGE_SELF)
    return dict(user_s=u.ru_utime, system_s=u.ru_stime, max_rss_bytes=u.ru_maxrss * 1024,
                minor_faults=u.ru_minflt, major_faults=u.ru_majflt, cycles=None, instructions=None,
                metric='Linux getrusage; ru_maxrss converted KiB to bytes; hardware counters unavailable')


def delta(after, before):
    return {key: (after[key] - before[key] if isinstance(after[key], (int, float))
                  and isinstance(before.get(key), (int, float)) else after[key]) for key in after}
