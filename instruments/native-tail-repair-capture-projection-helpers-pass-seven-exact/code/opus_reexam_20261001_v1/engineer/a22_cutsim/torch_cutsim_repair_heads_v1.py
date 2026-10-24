"""Pinned delegate for prospective separate repair heads; unqualified source only."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE.parent.parent / 'physicist/t87_torch_repair_heads_v1.py'
IMPLEMENTATION_SHA = 'ed1501930077f0267aa83674e0c975517513b3c766c9bcb9e105a3704a5404db'
_MODULE = None


def _load():
    global _MODULE
    assert hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest() == IMPLEMENTATION_SHA
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location('_t87_repair_heads_v1', IMPLEMENTATION)
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    return _MODULE


def sources():
    result = _load().sources()
    result[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def install(cd, opts=None):
    return _load().install(cd, opts)


def install_m(cd, original_M_install):
    return _load().install_m(cd, original_M_install)
