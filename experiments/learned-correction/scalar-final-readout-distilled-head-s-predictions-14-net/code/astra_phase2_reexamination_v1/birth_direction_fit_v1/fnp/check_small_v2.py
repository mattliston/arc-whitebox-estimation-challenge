"""Inert v2 launcher for the unchanged fixed n11 qualification gates.

Only core_v2.py and its source pin replace the original core selection.
The old checker, reference, fixtures, thresholds, output-dir CLI and result
schema remain unchanged.  Receipts bind the original checker and new core;
the caller must also preserve this launcher and build_v2.py.  No numerical
module is imported until the original explicit --execute entry point.
"""

import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / 'check_small.py'
CORE = HERE / 'core_v2.py'
ORIGINAL_SHA = '72ac54e903fcd068a940057d3dc5eeff1123973307154a4ee39a00c5e35154c5'
CORE_SHA = 'c31670320840545051ff7dc1031d71442cec99313423cff1e86ac62444b296f0'


def authenticate():
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise RuntimeError('Frozen v1 tiny checker changed')
    if hashlib.sha256(CORE.read_bytes()).hexdigest() != CORE_SHA:
        raise RuntimeError('Frozen FNP-left v2 core changed')


def main():
    authenticate()
    spec = importlib.util.spec_from_file_location('birth_direction_frozen_checker_v1', ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError('Frozen tiny checker import specification')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original_core = checker.CORE
    checker.CORE = CORE
    checker.PINS = {path: value for path, value in checker.PINS.items()
                    if path != original_core}
    checker.PINS[CORE] = CORE_SHA
    checker.main()
    authenticate()


if __name__ == '__main__':
    main()
