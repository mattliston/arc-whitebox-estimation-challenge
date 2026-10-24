"""radlab ptload: a copy of engineer/a54_jmiv/read_pt.py load() (torch zip checkpoints read with numpy, no torch)."""
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

