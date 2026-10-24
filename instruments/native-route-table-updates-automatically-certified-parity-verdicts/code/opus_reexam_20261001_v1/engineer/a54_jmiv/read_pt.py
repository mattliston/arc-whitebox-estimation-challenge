"""TAG opus_engineer a54 (2026-10-10, lead 21:02Z): read a trainer parameter file (.pt, torch zip format) WITHOUT torch: the pickle's
tensor rebuilds are mapped to numpy arrays over the zip's raw storages. Prints the per-layer shifts ('pl') and the dials, and with a
second file the largest difference of every array, dial and shift between the two.
usage: read_pt.py <a.pt> [<b.pt>]"""
import pickle
import sys
import zipfile
import numpy as np

DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64, 'IntStorage': np.int32, 'BoolStorage': np.bool_,
      'HalfStorage': np.float16}


def load(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    root = names[0].split('/')[0]
    class Storage:
        def __init__(self, kind):
            self.kind = kind
    def rebuild(storage, offset, size, stride, *rest):
        data, kind = storage
        a = np.frombuffer(data, dtype=DT[kind])
        if not size:
            return a[offset:offset + 1].reshape(()).copy()
        item = a.dtype.itemsize
        return np.lib.stride_tricks.as_strided(a[offset:], shape=tuple(size), strides=tuple(s * item for s in stride)).copy()
    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == '_rebuild_tensor_v2':
                return rebuild
            if name.endswith('Storage'):
                return name
            if module == 'collections' and name == 'OrderedDict':
                import collections
                return collections.OrderedDict
            if name == '_rebuild_parameter':
                return lambda data, requires_grad, hooks: data
            return super().find_class(module, name)
        def persistent_load(self, pid):
            _tag, kind, key, _loc, _n = pid
            kind = kind if isinstance(kind, str) else kind.__name__
            return (z.read(f'{root}/data/{key}'), kind)
    return U(z.open(f'{root}/data.pkl')).load()


a = load(sys.argv[1])
print('keys', sorted(a))
print('dials', {k: float(v) for k, v in a['dials'].items()})
for fam, v in sorted(a.get('pl', {}).items()):
    v = np.asarray(v, dtype=np.float64)
    if np.any(v):
        print('pl', fam, ' '.join('%d:%.6g' % (l, x) for l, x in enumerate(v) if x))
if len(sys.argv) > 2:
    b = load(sys.argv[2])
    for k in sorted(set(a['arrays']) | set(b['arrays'])):
        x, y = a['arrays'].get(k), b['arrays'].get(k)
        if x is None or y is None:
            print('array only in one file:', k); continue
        d = float(np.max(np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64))))
        if d:
            print('array', k, 'max diff %.3g' % d)
    for k in a['dials']:
        if float(a['dials'][k]) != float(b['dials'][k]):
            print('dial', k, float(a['dials'][k]), float(b['dials'][k]))
    for fam in sorted(set(a.get('pl', {})) | set(b.get('pl', {}))):
        x = np.asarray(a.get('pl', {}).get(fam, np.zeros(16)), dtype=np.float64)
        y = np.asarray(b.get('pl', {}).get(fam, np.zeros(16)), dtype=np.float64)
        if np.any(x != y):
            print('pl differs', fam, 'a:', ' '.join('%.6g' % t for t in x), '| b:', ' '.join('%.6g' % t for t in y))
