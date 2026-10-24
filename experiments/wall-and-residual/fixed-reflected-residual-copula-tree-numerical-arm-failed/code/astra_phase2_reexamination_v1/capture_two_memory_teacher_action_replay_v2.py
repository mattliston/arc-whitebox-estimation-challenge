"""Resource-only technical replay with bounded float64 CP reconstruction.

The original bank's arithmetic is unchanged. Only the offline post-tensor
read is accumulated in fixed2048-column chunks. Preserve v1 artifacts.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE/'capture_two_memory_teacher_action_v1.py'
PARENT_SHA = '4eb6a294651800c137614cbc1ad8a1bab99a206ad220d76e86f74b184056d608'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Unexpected frozen source substitution count: {old!r}')
    return source.replace(old, new)


def main():
    raw = PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen capture parent changed')
    source = raw.decode()
    source = replace_once(source, 'import resource\n', 'import resource\nimport os\n')
    source = replace_once(source, '"rows_two_memory_teacher_action_v1"',
                          '"rows_two_memory_teacher_action_v2"')
    source = replace_once(source,
        'source_names = list(EXPECTED) + [Path(__file__).name, DESIGN, TEST_PROTOCOL, TEST_PHYSICS]',
        'require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch at nice >=10")\n'
        '    source_names = list(EXPECTED) + [Path(__file__).name, DESIGN, TEST_PROTOCOL, TEST_PHYSICS,\n'
        '        "capture_two_memory_teacher_action_v1.py", "TWO_MEMORY_RESOURCE_REPLAY_V2.md"]')
    source = replace_once(source,
        '        p64 = p.astype(np.float64)\n'
        '        lam64 = lam.astype(np.float64)\n'
        '        cp = (p64*p64) @ (p64*lam64).T',
        '        lam64 = lam.astype(np.float64)\n'
        '        cp = np.zeros((p.shape[0], p.shape[0]), dtype=np.float64)\n'
        '        for begin in range(0, p.shape[1], 2048):\n'
        '            end = min(begin+2048, p.shape[1])\n'
        '            block = p[:, begin:end].astype(np.float64)\n'
        '            cp += (block*block) @ (block*lam64[begin:end]).T\n'
        '            del block')
    source = replace_once(source,
        '            terms = lam64*p64[i]*p64[i]*p64[j]',
        '            terms = (lam64*p[i].astype(np.float64)*p[i].astype(np.float64)\n'
        '                     *p[j].astype(np.float64))')
    source = replace_once(source,
        '        print(json.dumps(dict(rank=args.rank, captured_post_layer=l,',
        '        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 6*(1<<30),\n'
        '                "Fixed6GiB acquisition RSS qualification exceeded")\n'
        '        print(json.dumps(dict(rank=args.rank, captured_post_layer=l,')
    source = replace_once(source,
        '        artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size)',
        '        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 6*(1<<30),\n'
        '                "Fixed6GiB final RSS qualification exceeded")\n'
        '        artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size)')
    namespace = dict(__file__=str(Path(__file__).resolve()),
                     __name__='two_memory_resource_replay_v2_frozen_parent')
    exec(compile(source, str(PARENT)+'[resource-replay-v2]', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
