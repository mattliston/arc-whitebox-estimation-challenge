"""Identical fixed stream replay after v1's source-freeze failure.

Only artifact identity/provenance changes. No numerical or sampling change.
The original v1 failure/progress files remain immutable.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE / 'acquire_cross_depth_energy_v1.py'
PARENT_SHA256 = '5cc845a25c3c2b8a58ee62fd611e828f89fafe2e05e27a1e9b2a2083850cf0a7'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Frozen source substitution count differs: {old!r}')
    return source.replace(old, new)


def main():
    raw = PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA256:
        raise RuntimeError('Frozen acquisition parent changed')
    source = raw.decode()
    source = replace_once(source,
        "f'cross_depth_energy_r35_{args.stream}_v1.npz'",
        "f'cross_depth_energy_r35_{args.stream}_v2.npz'")
    source = replace_once(source,
        "schema='cross-depth-angular-energy-acquisition-v1'",
        "schema='cross-depth-angular-energy-acquisition-v2'")
    source = replace_once(source,
        "for path in (Path(__file__),HERE/'cross_depth_energy_statistics_v1.py',\n"
        "                     HERE/'judge_cross_depth_energy_v1.py'):",
        "for path in (Path(__file__),HERE/'acquire_cross_depth_energy_v1.py',\n"
        "                     HERE/'cross_depth_energy_statistics_v1.py',\n"
        "                     HERE/'judge_cross_depth_energy_v1.py',\n"
        "                     HERE/'judge_cross_depth_energy_replay_v2.py',\n"
        "                     HERE/'CROSS_DEPTH_ENERGY_TECHNICAL_REPLAY_V2.md'):")
    namespace = dict(__file__=str(Path(__file__).resolve()),
                     __name__='cross_depth_energy_replay_v2_frozen_parent')
    exec(compile(source, str(PARENT) + '[replay-v2-identities]', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
