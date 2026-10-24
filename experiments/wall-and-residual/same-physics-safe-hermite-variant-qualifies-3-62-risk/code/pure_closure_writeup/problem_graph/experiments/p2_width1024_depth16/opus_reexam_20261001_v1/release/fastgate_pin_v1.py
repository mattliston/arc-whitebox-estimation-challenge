"""TAG opus-release fastgate harness pin v1 (2026-10-12 17:40 UTC; lead 17:1xZ condition 3: the sha assertion and the boundary_run.py sha
pin are mandatory; any harness or flag change falls back to the full gate).
A count session can stand in for release's own stage-C / 1.3x slow session only if the SAME harness ran it with the SAME flags on the SAME file.
  --make OUT.json                         writes the pin (refuses to overwrite): sha256 of every harness file (boundary_run.py, boundary_client.py,
                                          setup_clock.py, server_light.py, rusage.py), a digest of the common.py values the harness imports
                                          (CLIENT_PY, SERVER_PY, THREADS, R, sha) and the client flopscope version of REFERENCE (a session dir).
  --check PIN FILE NORMAL SLOW [--n39 D]  prints PIN OK and exits 0 when (1) every harness file's sha256 equals the pin and the file is older than
                                          each session (the session ran on the pinned bytes; common.py, edited 2026-10-12 08:20Z for GUARD only,
                                          is held by its digest instead); (2) each session's manifest and launcher name FILE's full sha256;
                                          (3) flags: short_setup true, setup_slowdown 1.0 (NORMAL, n39) / 1.3 (SLOW), sites and profiled false,
                                          ranks [35, 100] ([35, 100, 39] for n39), flopscope as pinned; (4) client and server exit 0, and the guard
                                          record completed with the whole group retired. Else exits 1 listing every reason (fastgate_v1 then runs
                                          the full gate)."""
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
HARNESS = ['boundary_run.py', 'boundary_client.py', 'setup_clock.py', 'server_light.py', 'rusage.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def common_digest():
    sys.path.insert(0, str(R))
    import common
    text = '\n'.join([str(common.CLIENT_PY), str(common.SERVER_PY), json.dumps(common.THREADS, sort_keys=True), str(common.R),
                      inspect.getsource(common.sha)])
    return hashlib.sha256(text.encode()).hexdigest()


def check_session(d, digest, slowdown, ranks, pin, oldest_ok):
    d = Path(d)
    bad = []
    try:
        m = json.loads((d/'manifest.json').read_text())
        la = json.loads((d/'launcher.json').read_text())
        g = json.loads((d/'guard.json').read_text())
        (d/'rows.jsonl').read_text()
    except (OSError, ValueError) as e:
        return [f'{d.name}: unreadable session ({e})']
    born = os.stat(d).st_birthtime
    if born <= oldest_ok:
        bad.append(f'{d.name}: made before the newest harness file changed')
    if m.get('sha256') != digest or la.get('sha256') != digest:
        bad.append(f"{d.name}: ran sha {str(m.get('sha256'))[:12]} / {str(la.get('sha256'))[:12]}, not the file's {digest[:12]}")
    want = dict(short_setup=True, setup_slowdown=slowdown, sites=False, profiled=False, ranks=ranks, client_flopscope=pin['client_flopscope'])
    for k, v in want.items():
        if m.get(k) != v:
            bad.append(f'{d.name}: {k} = {m.get(k)!r}, needs {v!r}')
    if str(la.get('client_exit')) != '0' or str(la.get('server_exit')) != '0':
        bad.append(f"{d.name}: client exit {la.get('client_exit')}, server exit {la.get('server_exit')}")
    if not (g.get('completed') and g.get('whole_group_retired')):
        bad.append(f"{d.name}: guard record not completed ({g.get('error')})")
    return bad


def main():
    if sys.argv[1] == '--make':
        out, ref = Path(sys.argv[2]), Path(sys.argv[3])
        assert not out.exists(), f'{out} exists'
        pin = dict(made_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   harness={f: sha(R/f) for f in HARNESS}, common_digest=common_digest(),
                   client_flopscope=json.loads((ref/'manifest.json').read_text())['client_flopscope'])
        out.write_text(json.dumps(pin, indent=1) + '\n')
        print(json.dumps(pin, indent=1))
        return 0
    assert sys.argv[1] == '--check'
    pin = json.loads(Path(sys.argv[2]).read_text())
    file, normal, slow = Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    n39 = Path(sys.argv[sys.argv.index('--n39') + 1]) if '--n39' in sys.argv else None
    bad = [f'harness {f} changed (sha {sha(R/f)[:12]}, pin {h[:12]})' for f, h in pin['harness'].items() if sha(R/f) != h]
    if common_digest() != pin['common_digest']:
        bad.append('common.py values of the harness changed')
    newest = max(os.stat(R/f).st_mtime for f in pin['harness'])
    digest = sha(file)
    bad += check_session(normal, digest, 1.0, [35, 100], pin, newest)
    bad += check_session(slow, digest, 1.3, [35, 100], pin, newest)
    if n39:
        bad += check_session(n39, digest, 1.0, [35, 100, 39], pin, newest)
    if bad:
        print('PIN FAILED: ' + '; '.join(bad))
        return 1
    print(f"PIN OK: {file.name} sha {digest[:12]}; sessions {normal} / {slow}{' / ' + str(n39) if n39 else ''} ran the pinned harness "
          f"({', '.join(f[:-3] + ' ' + h[:8] for f, h in pin['harness'].items())}) with the gate's flags")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
