"""Harness-only correction: initialize the exact parent arena before kernels."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = (HERE/'unit.py').read_bytes()
assert hashlib.sha256(original).hexdigest() == '402de7c5d4af01ba679ceba74e0117241218879824a49b03bad1452368bf18ca'
source = original.decode()
before = '        spec.loader.exec_module(mod)\n'
after = '''        spec.loader.exec_module(mod)
        arena = fnp.empty((mod._PACKED_ELEMENTS,), dtype=fnp.float32)
        mod._SCRATCH = [arena, mod._cp3_bind_arena_views(arena, mod._PACKED_PUSH),
                        mod._cp3_bind_arena_views(arena, mod._PACKED_READ),
                        None, None, None, None]
        mod._bind_push6(mod._SCRATCH[0])
'''
assert source.count(before) == 1 and source.count("out = HERE/'unit_v1'") == 1
source = source.replace(before, after).replace("out = HERE/'unit_v1'", "out = HERE/'unit_v2'")
exec(compile(source, str(HERE/'unit.py'), 'exec'),
     {'__file__':__file__, '__name__':'__main__'})
