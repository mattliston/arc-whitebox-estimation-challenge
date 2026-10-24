"""Inert v2 builder wrapper: FNP array on the left of coefficient multiply.

The original builder, core, checker and results are preserved.  This imports
only the frozen stdlib-only builder and selects core_v2.py.  Its main,
coefficient rule, inverse proof and exclusive-output behavior are unchanged.
The generated build.json therefore binds the ORIGINAL builder and the NEW
core; qualification should also bind this wrapper's source/hash.  No
numerical module, candidate or backend-accounting object is imported.
"""

import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / 'build.py'
ORIGINAL_CORE = HERE / 'core.py'
CORE = HERE / 'core_v2.py'
ORIGINAL_SHA = '3569b34c713b6d7b92e713060cb1bcbc2a442c7363ad41c96bacabe58a9ebdf0'
ORIGINAL_CORE_SHA = 'd6c858751291d2305e4b1aa74fd828d9fa0c4f5fdd3009e239825324344431ee'
BEFORE = b'        change = change + c[j] * direction\n'
AFTER = b'        change = change + direction * c[j]\n'


def authenticate():
    builder = ORIGINAL.read_bytes()
    old = ORIGINAL_CORE.read_bytes()
    new = CORE.read_bytes()
    if hashlib.sha256(builder).hexdigest() != ORIGINAL_SHA:
        raise RuntimeError('Frozen v1 builder changed')
    if hashlib.sha256(old).hexdigest() != ORIGINAL_CORE_SHA:
        raise RuntimeError('Frozen v1 core changed')
    if old.count(BEFORE) != 1 or old.replace(BEFORE, AFTER, 1) != new:
        raise RuntimeError('V2 must contain only the FNP-left scalar multiplication edit')
    return hashlib.sha256(new).hexdigest()


def main():
    expected_core = authenticate()
    spec = importlib.util.spec_from_file_location('birth_direction_frozen_builder_v1', ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError('Frozen builder import specification')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.CORE = CORE
    builder.main()
    if authenticate() != expected_core:
        raise RuntimeError('V2 source changed during source generation')


if __name__ == '__main__':
    main()
