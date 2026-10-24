"""[opus-release] real-file release steps v2 (2026-10-16) = v1 with the diff rule generalized: the parity copy may differ from the real
file ONLY in governor threshold assignment lines (names matching ^_GOV_[A-Z0-9_]*(CAP|WALL)[A-Z0-9_]*_S = ), each set to 1e9 in the
parity copy. Everything else as v1.
[v1 doc] real-file release steps for a governor build whose boundary/official120 ran on an inert parity copy (v1, 2026-10-16).
The parity copy must differ from the real file in exactly one line (the governor threshold). Natively the governor is inert
(local flopscope has no _budget._active_context), so native replays of the REAL file must equal the parity copy's production bytes.
Steps (helpers from release_pipeline_v5.py; NEVER fires):
 A one-line diff check real vs parity copy
 B stage A on the REAL file (generic: proven = fired 2b108)
 C captured-setup parity on the REAL file: setup-only replays (tiny) + native replays normal/slow (sequential, guarded),
   bytes + FLOPs == the parity copy's production boundary rows
 D numerics provenance: the parity copy's official120 gates (must have passed) + byte/count identity vs S2tr's boundary
 E RELEASE_CONFIRMED (fire argv for the LEAD only)    F prefire vs the reference archive (CLEAR + 0 new signatures)
usage: release_gov_realfile_v1.py --source REAL.py --parity-rel REL_DIR --s2tr-boundary DIR --tag TAG --desc '...' [--science @list]"""
import argparse, importlib.util, json, re, subprocess, tarfile
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rp5', P/'release_pipeline_v5.py'); rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
O, REL, RUNS, PY, NPY = rp.O, rp.REL, rp.RUNS, rp.PY, rp.NPY
sha, read, put, env, utc, log, run, absent, admit = rp.sha, rp.read, rp.put, rp.env, rp.utc, rp.log, rp.run, rp.absent, rp.admit


def rows_of(d):
    return [json.loads(x) for x in (d/'rows.jsonl').read_text().splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=Path); ap.add_argument('--parity-rel', required=True, type=Path)
    ap.add_argument('--s2tr-boundary', required=True, type=Path); ap.add_argument('--tag', required=True); ap.add_argument('--desc', required=True)
    ap.add_argument('--science', default=''); ap.add_argument('--ref-archive', type=Path, default=rp.REF)
    a = ap.parse_args(); src = a.source.resolve(); h = sha(src); tag = a.tag; R = a.parity_rel.resolve(); B = R/'boundary'
    assert re.fullmatch(r'[a-z0-9_]+', tag), tag
    D = O/'lead/runs/opus_release_v1'/f'rel_{tag}'; D.mkdir()
    for p in (RUNS/f'a_{tag}', O/'fire'/tag):
        assert not p.exists(), f'tag already used: {p}'
    items = Path(a.science[1:]).read_text().split() if a.science.startswith('@') else [x for x in a.science.split(',') if x]
    science = [Path(x).resolve() for x in items]; assert all(x.is_file() for x in science)
    start = read(R/'START_v1.json'); par_src = Path(start['source']); assert sha(par_src) == start['source_sha256']
    put(D/'START_v1.json', dict(utc=utc(), source=str(src), source_sha256=h, tag=tag, desc=a.desc, parity_rel=str(R), parity_source=str(par_src),
                                parity_source_sha256=start['source_sha256'], science={str(x): sha(x) for x in science}, script_sha256=sha(__file__)))
    log(D, f'START {tag} real {h[:12]} parity copy {start["source_sha256"][:12]}')
    # A one-line diff
    a_lines, b_lines = par_src.read_text().splitlines(), src.read_text().splitlines()
    assert len(a_lines) == len(b_lines), 'line counts differ'
    pat = re.compile(r'^(_GOV_[A-Z0-9_]*(?:CAP|WALL)[A-Z0-9_]*_S) = (.+?)(\s*#.*)?$')
    diff = []
    for x, y in zip(a_lines, b_lines):
        if x != y:
            mx, my = pat.match(x), pat.match(y)
            assert mx and my and mx.group(1) == my.group(1) and float(mx.group(2)) == 1e9, (x, y)
            diff.append((mx.group(1), mx.group(2), my.group(2)))
    assert diff, 'no governor threshold differs'
    log(D, f'STEPA diff vs parity copy = governor thresholds only (parity 1e9 -> real): {diff}')
    # B stage A on the real file
    r = run([PY, '-B', str(P/'stage_a_generic_v1.py'), str(src), f'a_{tag}'], D, 'STEPB_stage_a', cwd=O)
    A = RUNS/f'a_{tag}'; st = read(A/'stage_a.json'); arc = A/'submission.tar.gz'
    log(D, f"STEPB stage A passed={st['passed']} {st['line']}")
    assert st['passed'] and st['candidate_sha256'] == h
    with tarfile.open(arc) as f: assert sorted(f.getnames()) == ['estimator.py', 'manifest.json'] and f.extractfile('estimator.py').read() == src.read_bytes()
    # C captured-setup parity of the REAL file against the parity copy's production boundary
    N = D/'parity'; N.mkdir()
    for m in ('normal', 'slow'):
        rr = run(['nice', '-n', '10', NPY, '-B', str(P/'native_probe_generic_v1.py'), str(src), '35,100', str(N/f'tiny_{m}')], D, f'STEPC_tiny_{m}',
                 e=env(SETUP_ONLY='1', CAPTURED_SETUP_MANIFEST=str(B/m/'manifest.json'), CANDIDATE_SHA256=h))
        assert rr.returncode == 0 and 'CAPTURED_SETUP_REPLAY_PASS' in rr.stdout, rr.stdout[-1500:] + rr.stderr[-1500:]
    log(D, 'STEPC setup-only replays PASS on the real file (all semantic setup phases exact)')
    pids, rows = [], []
    for m in ('normal', 'slow'):
        admit(D, 1)
        cmd = [PY, '-B', str(REL/'guard_footprint_v3.py'), '--record', str(N/f'guard_{m}.json'), '--seconds', '1100', '--footprint-limit', '7500000000',
               '--rss-backstop', '10000000000', 'nice', '-n', '10', NPY, '-B', str(P/'native_probe_generic_v1.py'), str(src), '35,100', str(N/m)]
        p = subprocess.Popen(cmd, cwd=REL, env=env(CAPTURED_SETUP_MANIFEST=str(B/m/'manifest.json'), CANDIDATE_SHA256=h),
                             stdout=open(N/f'{m}.log', 'x'), stderr=subprocess.STDOUT)
        assert p.wait() == 0, m; pids.append(p.pid)
        g = read(N/f'guard_{m}.json'); pids.append(g['pid']); assert g['completed'] and g['exit_code'] == 0 and g['whole_group_retired']
        nm = read(N/m/'manifest.json'); assert nm['sha256'] == h and nm['setup']['captured_replay']['all_setup_counts_exact']
        for x, y in zip(rows_of(N/m), rows_of(B/m)):
            k = x['rank']; assert k == y['rank']
            nat = np.load(N/m/f'pred_{k}.npy'); prod = np.frombuffer((B/m/f'pred_{k}.bin').read_bytes(), dtype=np.float32).reshape(16, 1024).astype(np.float64)
            rows.append(dict(mode=m, rank=k, bytes_equal=bool(nat.tobytes() == prod.tobytes()), native_flops=x['flops'], production_flops=y['flops']))
    assert absent(pids) and [r_['rank'] for r_ in rows] == [35, 100, 35, 100]
    par = dict(passed=all(r_['bytes_equal'] and r_['native_flops'] == r_['production_flops'] for r_ in rows), candidate_sha256=h,
               production_from=str(B), production_source_sha256=start['source_sha256'], rows=rows, owned_ids=pids, all_retired=True)
    put(D/'CAPTURED_SETUP_PARITY_v1.json', par); log(D, f"STEPC parity passed={par['passed']} {rows}")
    assert par['passed']
    # D numerics provenance
    gate = read(R/'RELEASE_GATE_OFFICIAL120_v1.json'); early = {x: read(B/x/'early_gate.json') for x in ('OFFICIAL_NORMAL_v1', 'OFFICIAL_SLOW_v1')}
    S = a.s2tr_boundary.resolve(); ident = {}
    for m in ('normal', 'slow'):
        for x, y in zip(rows_of(B/m), rows_of(S/m)):
            k = x['rank']; assert k == y['rank']
            ident[f'{m}_{k}'] = dict(pred_bytes_equal=(B/m/f'pred_{k}.bin').read_bytes() == (S/m/f'pred_{k}.bin').read_bytes(),
                                     **{f: [x.get(f), y.get(f)] for f in ('requests', 'flops', 'free_trips') if f in x or f in y})
    same = all(v['pred_bytes_equal'] and all(p_[0] == p_[1] for f, p_ in v.items() if f != 'pred_bytes_equal') for v in ident.values())
    prov = dict(official120_from=f'parity copy {par_src.name} {start["source_sha256"][:8]} (differs from the real file only in the governor thresholds {diff}); its boundary/early gates/stage C/designation judge/release gate ran in {R}',
                parity_release_gate_passed=bool(gate.get('passed')), early_gates={k: v.get('passed', v.get('status')) for k, v in early.items()},
                s2tr_boundary=str(S), parity_boundary_vs_s2tr=ident, parity_boundary_identical_to_s2tr=same,
                statement=('S2tr official120 numerics apply byte-for-byte: the parity copy boundary rows (requests, FLOPs, free trips, prediction bytes) equal S2tr\'s on every rank and mode; '
                           'the real file equals the parity copy natively (STEPC) and on the untriggered production path by construction (one-literal diff).') if same else
                          'parity copy boundary differs from S2tr somewhere (see parity_boundary_vs_s2tr): only the parity copy\'s own official120 numbers apply')
    put(D/'NUMERICS_PROVENANCE_v1.json', prov); log(D, f"STEPD provenance: parity gate passed={prov['parity_release_gate_passed']} identical_to_s2tr={same}")
    assert prov['parity_release_gate_passed']
    # E confirmation
    ev = [src, par_src, arc, A/'stage_a.json', D/'CAPTURED_SETUP_PARITY_v1.json', D/'NUMERICS_PROVENANCE_v1.json', R/'RELEASE_GATE_OFFICIAL120_v1.json',
          R/'RELEASE_CONFIRMED_v1.json', R/'COLLECTORS_RETIREMENT_v1.json', rp.POLICY] + science + [B/x/'early_gate.json' for x in early]
    argv = [PY, '-B', str(O/'fire/fire_once.py'), tag, str(arc), a.desc]
    put(D/'RELEASE_CONFIRMED_v1.json', dict(status='CONFIRMED', utc=utc(), executed_by='opus-release', candidate_path=str(src), candidate_sha256=h,
        archive_path=str(arc), archive_sha256=sha(arc), numerics_provenance=prov, official_wall_policy=dict(path=str(rp.POLICY), sha256=sha(rp.POLICY)),
        fire_argv_for_lead_only=argv, fire_cwd=str(O), submitted=False, evidence_sha256={str(x): sha(x) for x in ev}))
    log(D, f'STEPE CONFIRMED confirmation {sha(D/"RELEASE_CONFIRMED_v1.json")[:12]} archive {sha(arc)[:12]}')
    # F prefire
    rr = run([PY, '-B', str(O/'fire/prefire_check_v1.py'), str(arc), h[:8], str(a.ref_archive)], D, 'STEPF_prefire', cwd=O)
    tail = [l for l in rr.stdout.splitlines() if not l.startswith('  eye')][-3:]
    log(D, f'STEPF prefire rc={rr.returncode} ' + ' | '.join(tail))
    clear = rr.returncode == 0 and 'PREFIRE CLEAR | new signatures: 0' in rr.stdout
    put(D/'PIPELINE_DONE_v1.json', dict(utc=utc(), status='CONFIRMED', prefire_clear_zero_new=clear,
        fire_command=(' '.join(argv[:-1]) + f" '{a.desc}'") if clear else None))
    log(D, f'DONE status=CONFIRMED prefire_clear={clear}' + (f"\nFIRE (lead only, cwd O): {' '.join(argv[:-1])} '{a.desc}'" if clear else ''))


if __name__ == '__main__':
    main()
