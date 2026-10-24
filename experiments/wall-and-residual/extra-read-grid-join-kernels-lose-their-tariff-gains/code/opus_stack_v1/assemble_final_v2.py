"""[opus-stack] final-stack assembler v2 = v1 + --lazy-last (append the lazybind v2 fragment AFTER every --frag, the lead's order
v4f -> prune -> pushSM -> join/oldbank SM -> lazybind v2).
[v1 doc] final-stack assembler v1. Builds the RELEASE file and its PARITY COPY from one governor base (GOV v4f / v4g release file,
which already carries S2tr) by the same deterministic steps, in this order:
  1 dm2: the three asserted single-occurrence edits of live defs (lead/runs/opus_exact_v1/dm2/build_dm2_v1.py), byte-identical
  2 lazybind v2 fragment (lead/runs/opus_lazybind_v1/fragment_lazybind_v2.py.txt), byte-identical
  3 every --frag file appended in the given order (pushSM, join/oldbank SM, ...); --editor files are python modules with apply(text)->text
    run in the given order AFTER the fragments (for lanes that hand over asserted edits instead of fragments)
The parity copy = the release file with EVERY governor threshold assignment (^_GOV_..CAP.._S / ^_GOV_..WALL.._S at column 0) set to
1e9 (asserted: at least one, each assignment line unique), so boundary/parity counts never depend on the local machine's residual/wall.
Checks: both compile; release vs parity differ only in those lines; writes ASSEMBLY_<tag>.json with every input's sha256.
usage: assemble_final_v1.py <tag> --gov <release.py> <sha8> [--no-dm2] [--no-lazy] [--frag f ...] [--editor e.py ...]"""
import hashlib, importlib.util, json, re, sys, time
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()
THRESH = re.compile(r'^(_GOV_[A-Z0-9_]*(?:CAP|WALL)[A-Z0-9_]*_S)( *= *)([0-9.eE+-]+)(.*)$')


def dm2(text):
    code = (O/'lead/runs/opus_exact_v1/dm2/build_dm2_v1.py').read_text()
    body = code[code.index('def sub('):code.index('Path(sys.argv[3]).write_text(s)')]
    ns = dict(base=text)
    exec(compile(body, 'dm2', 'exec'), ns)
    return ns['s']


def parity_of(text):
    out, hits = [], []
    for line in text.split('\n'):
        m = THRESH.match(line)
        if m:
            hits.append((m.group(1), m.group(3)))
            line = m.group(1) + m.group(2) + '1e9' + m.group(4)
        out.append(line)
    names = [h[0] for h in hits]
    assert hits and len(names) == len(set(names)), hits
    return '\n'.join(out), hits


def main():
    a = sys.argv[1:]; tag = a[0]; assert re.fullmatch(r'[a-z0-9_]+', tag)
    gi = a.index('--gov'); gov, gsha = Path(a[gi + 1]), a[gi + 2]
    frags = [Path(a[i + 1]) for i, x in enumerate(a) if x == '--frag']
    editors = [Path(a[i + 1]) for i, x in enumerate(a) if x == '--editor']
    base = gov.read_text(); assert sha(base).startswith(gsha), 'governor base sha'
    steps, src = [dict(step='base', path=str(gov), sha256=sha(base))], base
    if '--no-dm2' not in a:
        src = dm2(src); steps.append(dict(step='dm2', script_sha256=sha((O/'lead/runs/opus_exact_v1/dm2/build_dm2_v1.py').read_bytes())))
    lz = (O/'lead/runs/opus_lazybind_v1/fragment_lazybind_v2.py.txt').read_text()
    if '--no-lazy' not in a and '--lazy-last' not in a:
        src += lz; steps.append(dict(step='lazybind_v2', sha256=sha(lz)))
    for f in frags:
        t = f.read_text(); src += ('' if t.startswith('\n') else '\n\n') + t; steps.append(dict(step='frag', path=str(f), sha256=sha(t)))
    if '--no-lazy' not in a and '--lazy-last' in a:
        src += lz; steps.append(dict(step='lazybind_v2', sha256=sha(lz)))
    for e in editors:
        spec = importlib.util.spec_from_file_location('editor', e); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        src = mod.apply(src); steps.append(dict(step='editor', path=str(e), sha256=sha(e.read_bytes())))
    par, hits = parity_of(src)
    compile(src, 'release', 'exec'); compile(par, 'parity', 'exec')
    a_l, b_l = src.split('\n'), par.split('\n'); assert len(a_l) == len(b_l)
    differ = [x for x, y in zip(a_l, b_l) if x != y]
    assert len(differ) == len(hits) and all(THRESH.match(x) for x in differ)
    rel, pty = D/f'cand_{tag}_release.py', D/f'cand_{tag}_parity.py'
    assert not rel.exists() and not pty.exists(), 'tag used'
    rel.write_text(src); pty.write_text(par)
    man = dict(tag=tag, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), steps=steps, release=str(rel), release_sha256=sha(src),
               parity=str(pty), parity_sha256=sha(par), governor_thresholds_release=hits, differing_lines=differ, bytes=len(src))
    (D/f'ASSEMBLY_{tag}.json').write_text(json.dumps(man, indent=1) + '\n')
    print(json.dumps(dict(release=man['release_sha256'][:12], parity=man['parity_sha256'][:12], thresholds=hits, steps=[s['step'] for s in steps])))


if __name__ == '__main__':
    main()
