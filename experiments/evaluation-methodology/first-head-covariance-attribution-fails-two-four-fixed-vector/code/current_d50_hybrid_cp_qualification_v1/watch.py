"""Reuse frozen independent watchdog with only new paths and stage roster."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'current_bd_p_coadaptation_v1/watch.py'
EXPECTED = 'b51590c06abf4394ed78274756e99046de9bccc2240dd55e8543f64e83e7304b'


def main():
    original = SOURCE.read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED:
        raise RuntimeError('Frozen resource supervisor changed')
    source = original.decode()
    replacements = (
        ("STAGES = ('cpu_forward', 'mps_forward', 'cpu_gradient', 'mps_gradient', 'judge', 'train')",
         "STAGES = ('mps_forward', 'cpu_gradient', 'mps_gradient')"),
        ("HERE/'PROTOCOL_V1.md'", "HERE/'PROTOCOL.md'"),
        ("schema='current_d50_p_coadaptation_watch_v1'", "schema='current_d50_hybrid_cp_watch_v1'"),
    )
    for old,new in replacements:
        if source.count(old)!=1:
            raise RuntimeError('Unexpected supervisor source span')
        source = source.replace(old,new)
    inverse = source
    for old,new in reversed(replacements):
        inverse = inverse.replace(new,old)
    if inverse != original.decode():
        raise RuntimeError('Supervisor inverse differs')
    namespace = dict(__name__='hybrid_cp_watch_frozen_body',__file__=str(Path(__file__).resolve()))
    exec(compile(source,str(__file__)+':frozen_watch','exec'),namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
