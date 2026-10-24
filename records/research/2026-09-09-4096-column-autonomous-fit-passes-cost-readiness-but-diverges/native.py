"""Thin fixed-two-net wrapper over the preserved official-input native harness.

Inert without --source. The pinned harness's input, budget, materialization,
resource, repeat and post-bill log-reset loop is reused without rewriting it.
Only cap build/prerequisite admission, small observer receipts and the fixed
360B cost gate are added. No target access and no estimator arithmetic changes.
"""
import ast
import json
from pathlib import Path
import sys
from common import BD, HERE, NEW, PARENT, PARENT_SHA, PINS, bindings, completed, module, sha, unchanged


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Frozen harness span mismatch: '+old[:100])
    return source.replace(old, new)


def verify_build(candidate, inputs, tiny_sha):
    assert candidate == (HERE/'estimator.py').resolve()
    receipt = json.loads((HERE/'build.json').read_text())
    assert receipt['passed'] and receipt['inverse_source_byte_exact']
    assert receipt['all16_P_unchanged'] and receipt['signed_pair_cap'] == 4096
    assert receipt['parent_sha256'] == PARENT_SHA and receipt['candidate_sha256'] == sha(candidate)
    unchanged(receipt['source_bindings'])
    inputs.update(receipt['source_bindings'])
    tiny = completed(HERE/'tiny', tiny_sha)
    assert tiny['schema'] == 'current_paircap4096_e2e_tiny_v1'
    assert tiny['source_bindings'][str(HERE/'model.py')] == sha(HERE/'model.py')
    for path in (HERE/'build.json', HERE/'tiny/result.json', HERE/'tiny/completion.json'):
        inputs[str(path)] = sha(path)


class Observer:
    """Retain only already-computed selector outputs; materialize after Budget."""
    def __init__(self, candidate):
        self.selects, self.energies = [], []
        old_select, old_energy = candidate.select, candidate._paircap_energy
        def selected(*args, **kwargs):
            answer = old_select(*args, **kwargs)
            self.selects.append(answer[0])
            return answer
        def energy(*args, **kwargs):
            answer = old_energy(*args, **kwargs)
            self.energies.append(answer)
            return answer
        candidate.select, candidate._paircap_energy = selected, energy

    def clear(self):
        self.selects.clear()
        self.energies.clear()

    def materialize(self, np):
        choices = []
        for value in self.energies:
            e = np.asarray(value, dtype=np.float64)
            assert e.ndim == 1 and len(e) > 4096 and np.isfinite(e).all()
            order = np.argsort(-e, kind='stable')
            boundary = float(e[order[4095]])
            choices.append(dict(pairs_before=len(e), chosen=np.sort(order[:4096]).tolist(),
                minimum_relative_margin=(boundary-float(e[order[4096]]))/max(abs(boundary), 1e-300)))
        return dict(selections=[np.asarray(x, dtype=np.int64).tolist() for x in self.selects],
            retirement=choices, ownership='Already computed native values, copied only after participant context')


def main():
    if '--source' not in sys.argv:
        print('Inert. --source estimator.py --split development --out native --tiny-sha256 SHA')
        return 0
    assert sha(BD/'native_v2.py') == PINS[BD/'native_v2.py']
    old = module(BD/'native_v2.py', 'cap4096_preserved_native_harness')
    text = (BD/'native_v2.py').read_text()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    source = ''.join(text.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno])
    source = once(source, "    parser.add_argument('--source',type=Path)",
        "    parser.add_argument('--source',type=Path)\n    parser.add_argument('--tiny-sha256')")
    start = "    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',\n        HERE/'NATIVE_SCOPE_CLARIFICATION.md',Path(__file__),STORE/'manifest.json')}"
    source = once(source, start,
        "    inputs=bindings((candidate,HERE/'PROTOCOL.md',HERE/'IMPLEMENTATION.md',HERE/'common.py',Path(__file__)))")
    start = source.index('    if candidate==BANK:\n')
    stop = source.index('    manifest=json.loads(', start)
    source = source[:start] + (
        "    verify_build(candidate,inputs,args.tiny_sha256)\n"
        "    assert args.split=='development', 'Only fixed development35/100 is admitted'\n"
        "    ranks=(35,100)\n") + source[stop:]
    source = once(source, '        estimator=None\n', '        observer=Observer(module)\n        estimator=None\n')
    source = once(source, '                started=time.monotonic()\n',
        '                observer.clear()\n                started=time.monotonic()\n')
    source = once(source, "                trial['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)",
        "                diagnostic=args.out/f'call_{call:02d}_state.json'\n"
        "                write(diagnostic,observer.materialize(np))\n"
        "                result['artifacts'][str(diagnostic)]=sha(diagnostic)\n"
        "                trial['state_diagnostic']=str(diagnostic)\n"
        "                observer.clear()\n"
        "                trial['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)")
    source = once(source, '        result.update(passed=True,hashes_unchanged=True,',
        "        result['cost_gate_passed']=all(t['flops']<=360000000000 for t in result['trials'][:2])\n"
        "        result['warm_rank35_flops']=result['trials'][-1]['flops']\n"
        "        assert result['cost_gate_passed'], 'Fixed complete cost gate: both nets <=360B'\n"
        '        result.update(passed=True,hashes_unchanged=True,')
    env = dict(vars(old), __file__=str(Path(__file__).resolve()), HERE=HERE, PARENT=NEW,
        bindings=bindings, verify_build=verify_build, Observer=Observer)
    exec(compile(source, str(__file__)+':preserved_main', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
