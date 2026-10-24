"""TAG opus-lead prefire_check v1 (2026-10-12): read-only check of a release archive before fire/fire_once.py.
usage (from O): python -B fire/prefire_check_v1.py <archive.tar.gz> <expected estimator sha8> <reference archive = last fired file of the pair>
Checks: members = estimator.py + manifest.json; estimator sha256 starts with the expected sha8 and matches the manifest; imports within
ALLOWED; no exec / eval / compile / __import__ / numpy / ctypes / threading / subprocess / pickle / marshal / importlib (lines with 'torch'
or 'os.' are printed for the eye: comments only); einsum subscript set vs the reference (new signatures are printed: inspect each one and its
operands by hand before firing). Exit 0 = clear, 1 = a check failed. Extracts into memory only; writes nothing."""
import hashlib
import io
import json
import re
import sys
import tarfile

ALLOWED = {'base64', 'flopscope', 'flopscope.numpy', 'math', 'time', 'whestbench', 'zlib'}
BANNED = ('exec(', 'eval(', 'compile(', '__import__', 'ctypes', 'subprocess', 'threading', 'multiprocessing', 'import numpy',
          'from numpy', 'pickle', 'marshal', 'importlib', 'socket')


def load(path):
    raw = open(path, 'rb').read()
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
    members = sorted(m.name for m in tf.getmembers())
    files = {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
    return hashlib.sha256(raw).hexdigest(), members, files


def sigs(src):
    return set(re.findall(r"""einsum\(\s*['"]([^'"]*)['"]""", src))


arch, sha8, ref = sys.argv[1], sys.argv[2], sys.argv[3]
ok = True
a_sha, members, files = load(arch)
print('archive', arch, 'sha256', a_sha, 'bytes', sum(len(v) for v in files.values()))
if members != ['estimator.py', 'manifest.json']:
    ok = False; print('FAIL members', members)
est = files['estimator.py']; src = est.decode()
e_sha = hashlib.sha256(est).hexdigest()
man = json.loads(files['manifest.json'])
m_sha = [f['sha256'] for f in man['files'] if f['name'] == 'estimator.py']
print('estimator sha256', e_sha, 'lines', len(src.splitlines()), '| manifest', m_sha, '| entrypoint', man['entrypoint'])
if not e_sha.startswith(sha8) or m_sha != [e_sha]:
    ok = False; print('FAIL estimator sha vs expected %s / manifest' % sha8)
imps = set(re.findall(r'^\s*(?:import|from)\s+([\w.]+)', src, re.M))
print('imports', sorted(imps))
if not imps <= ALLOWED:
    ok = False; print('FAIL imports outside the allowed set:', sorted(imps - ALLOWED))
for w in BANNED:
    if w in src:
        ok = False; print('FAIL banned token %r x%d' % (w, src.count(w)))
for i, line in enumerate(src.splitlines(), 1):
    if 'torch' in line or re.search(r'(?<![\w.])os\.', line):
        print('  eye L%d: %s' % (i, line.strip()[:160]))
_, _, rfiles = load(ref)
s_new, s_ref = sigs(src), sigs(rfiles['estimator.py'].decode())
print('einsum calls %d, signatures %d (reference %d)' % (src.count('einsum('), len(s_new), len(s_ref)))
for s in sorted(s_new - s_ref):
    print('  NEW signature %r: inspect it and its operands before firing' % s)
    for i, line in enumerate(src.splitlines(), 1):
        if ("'%s'" % s) in line or ('"%s"' % s) in line:
            print('    L%d: %s' % (i, line.strip()[:200]))
for s in sorted(s_ref - s_new):
    print('  dropped signature %r' % s)
print('PREFIRE', 'CLEAR' if ok else 'FAILED', '| new signatures: %d' % len(s_new - s_ref))
sys.exit(0 if ok else 1)
