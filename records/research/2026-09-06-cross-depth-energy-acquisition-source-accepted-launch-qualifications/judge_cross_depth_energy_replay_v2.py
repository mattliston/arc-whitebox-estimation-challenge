"""Frozen v1 judge arithmetic, bound exclusively to qualified v2 replays."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE / 'judge_cross_depth_energy_v1.py'
PARENT_SHA256 = 'f43a7d2c0c3e35b6324ae9aed684d18e03fcc629e7f936075cf26ccaa85e974f'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Frozen source substitution count differs: {old!r}')
    return source.replace(old, new)


def main():
    raw = PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA256:
        raise RuntimeError('Frozen judge parent changed')
    source = raw.decode()
    for old, new in (
        ("f'cross_depth_energy_r35_{label}_v1.npz'",
         "f'cross_depth_energy_r35_{label}_v2.npz'"),
        ("'cross-depth-angular-energy-acquisition-v1'",
         "'cross-depth-angular-energy-acquisition-v2'"),
        ("'cross_depth_energy_judgment_v1.npz'",
         "'cross_depth_energy_judgment_v2.npz'"),
        ("schema='cross-depth-angular-energy-judgment-v1'",
         "schema='cross-depth-angular-energy-judgment-v2'"),
        ("for name in ('cross_depth_energy_statistics_v1.py','acquire_cross_depth_energy_v1.py',Path(__file__).name):",
         "for name in ('cross_depth_energy_statistics_v1.py','acquire_cross_depth_energy_v1.py',"
         "'judge_cross_depth_energy_v1.py','acquire_cross_depth_energy_replay_v2.py',"
         "'CROSS_DEPTH_ENERGY_TECHNICAL_REPLAY_V2.md',Path(__file__).name):"),
    ):
        source = replace_once(source, old, new)
    namespace = dict(__file__=str(Path(__file__).resolve()),
                     __name__='cross_depth_energy_judge_v2_frozen_parent')
    exec(compile(source, str(PARENT) + '[replay-v2-identities]', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
