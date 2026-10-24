"""[opus-release] end-to-end release pipeline v5 (--safe: first-priced SAFE gating) (sequential natives: boundary and parity one at a time) (RSS backstop 10 GB = release/common LIMITS; footprint 7.5 GB unchanged) (--science accepts @listfile and directories) (templated from the b709 release of 2026-10-16). Metadata + guarded natives; NEVER fires.
usage: release_pipeline_v1.py --source S.py --tag TAG --desc 'one sentence' [--science a.json,b.json] [--ref-archive A.tar.gz]
                              [--dry-run] [--ignore-gate-fail]
steps -> O/lead/runs/opus_release_v1/rel_<TAG>/ (+ O/release/runs/{a_,c_,fg_}<TAG>...):
 1 boundary normal(--short-setup)/slow(--setup-slowdown 1.3), ranks 35,100, guard v3, 2 natives   (~2 min)
 2 official120 early gates normal -> slow (P(loss)<=20%, binding, setup 4.5, worst 120, slow req +2%, slow setup 4.6)
 3 collectors retirement (owned PID/PPID/PGID absent)
 4 stage A archive (generic: proven = fired 2b108; exact estimator bytes; einsum/repeat/numpy/py3.10 checks)
 5 official120 metadata finalization (fastgate pin, stage_c x2, designation judge, release gate)
 6 captured-setup native parity: setup-only proofs, then native replays normal+slow (2 natives), bytes + FLOPs == production
 7 RELEASE_CONFIRMED (pins everything + --science evidence; fire argv for the LEAD only)
 8 prefire_check vs the reference archive (#336342 8153e9df by default): must be CLEAR with 0 new signatures"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tarfile, time
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
P = Path(__file__).resolve().parent; REL = O/'release'; RUNS = REL/'runs'
PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'; NPY = str(O.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
OFF = O/'lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1'
POLICY = O/'lead/ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json'
REF = REL/'runs/a_smirnov_rpe_20261015_astra_v1/submission.tar.gz'
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
def put(p, x):
    with Path(p).open('x') as f: f.write(x if isinstance(x, str) else json.dumps(x, indent=2, allow_nan=False) + '\n')
def env(**extra):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{k: '1' for k in THREADS})
    for k in ('PB_ALL','MODE','DRY','STATS','SETUP_ONLY','CAPTURED_SETUP_MANIFEST'): e.pop(k, None)
    e.update(extra); return e
def utc(): return subprocess.check_output(['/bin/date','-u','+%FT%TZ'], text=True).strip()
def log(D, msg):
    line = f"{utc()} {msg}"; print(line, flush=True)
    with (D/'PIPELINE.log').open('a') as f: f.write(line + '\n')
def free_pct(): return int(re.search(r'free percentage: (\d+)%', subprocess.check_output(['/usr/bin/memory_pressure'], text=True)).group(1))
def heavy(): return sum(int(x.split(None,1)[0]) >= 1000000 and 'python' in x.lower() for x in subprocess.check_output(['/bin/ps','-axo','rss=,comm='], text=True).splitlines() if x.strip())
def admit(D, need):
    for _ in range(360):
        f, h = free_pct(), heavy()
        if f >= 60 and h + need <= 7: return dict(free=f, heavy=h)
        time.sleep(5)
    raise SystemExit(f'admission timeout free={f} heavy={h}')
def run(cmd, D, name, cwd=REL, e=None, timeout=900):
    r = subprocess.run(cmd, cwd=cwd, env=e or env(), capture_output=True, text=True, timeout=timeout)
    put(D/f'{name}.log', ' '.join(map(str, cmd)) + f'\n# exit {r.returncode}\n' + r.stdout + r.stderr); return r
def absent(ids):
    live = [tuple(map(int, x.split())) for x in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='], text=True).splitlines() if x.strip()]
    return not [x for x in live if set(x) & set(ids)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=Path); ap.add_argument('--tag', required=True); ap.add_argument('--desc', required=True)
    ap.add_argument('--science', default=''); ap.add_argument('--ref-archive', type=Path, default=REF)
    ap.add_argument('--reviewed', default='', help='einsum signatures read by hand, ;-separated (stage A pass-through)')
    ap.add_argument('--safe', action='store_true', help='SAFE designation: skip primary-only early gates; judge + release gate --first-priced')
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--ignore-gate-fail', action='store_true')
    a = ap.parse_args(); src = a.source.resolve(); h = sha(src); tag = a.tag
    assert re.fullmatch(r'[a-z0-9_]+', tag), tag
    D = O/'lead/runs/opus_release_v1'/f'rel_{tag}'; D.mkdir()
    for p in (RUNS/f'a_{tag}', RUNS/f'c_{tag}_official120', RUNS/f'{tag}_official120_slow', RUNS/f'fg_{tag}_official120_n_src', RUNS/f'fg_{tag}_official120_s_src', O/'fire'/tag):
        assert not p.exists(), f'tag already used: {p}'
    items = Path(a.science[1:]).read_text().split() if a.science.startswith('@') else [x for x in a.science.split(',') if x]
    science = []
    for x in items:
        x = Path(x).resolve(); assert x.exists(), f'missing evidence {x}'
        science += sorted(y for y in x.rglob('*') if y.is_file()) if x.is_dir() else [x]
    put(D/'START_v1.json', dict(utc=utc(), source=str(src), source_sha256=h, tag=tag, desc=a.desc, science={str(x): sha(x) for x in science},
                                ref_archive=str(a.ref_archive), ref_archive_sha256=sha(a.ref_archive), dry_run=a.dry_run, pipeline_sha256=sha(__file__)))
    log(D, f'START {tag} source {h[:12]}')
    gate_fail = []
    # 1 boundary
    B = D/'boundary'
    r = run(['bash', str(P/'run_boundary_v5.sh'), str(src), str(B)], D, 'STEP1_boundary', cwd=P, timeout=3600)
    ex = read(B/'EXITS_v1.json'); assert ex['normal_exit'] == 0 and ex['slow_exit'] == 0, ex
    log(D, f"STEP1 boundary exits normal={ex['normal_exit']} slow={ex['slow_exit']}")
    # 2 early gates (primary only; a SAFE is judged --first-priced in step 5)
    eg = OFF/'early_count_gate_official120_v1.py'
    if a.safe: log(D, 'STEP2 SAFE mode: primary early gates skipped; step 5 judges --first-priced (P(loss) first priced <= 10%, warm <= 42,700)')
    if not a.safe:
        rn = run([PY, '-B', str(eg), str(src), str(B/'normal'), 'normal', str(B/'OFFICIAL_NORMAL_v1')], D, 'STEP2_gate_normal')
        lines = (B/'OFFICIAL_NORMAL_v1/judge.txt').read_text() if (B/'OFFICIAL_NORMAL_v1/judge.txt').exists() else rn.stdout + rn.stderr
        log(D, 'STEP2 normal gate rc=%d\n%s' % (rn.returncode, '\n'.join(l for l in lines.splitlines() if l.startswith(('  ok','  FAIL')))))
        if rn.returncode != 0:
            gate_fail.append('normal')
            if not a.ignore_gate_fail: raise SystemExit('STOP: normal early gate failed (see judge lines)')
        if rn.returncode == 0:
            rs = run([PY, '-B', str(eg), str(src), str(B/'slow'), 'slow', str(B/'OFFICIAL_SLOW_v1'), str(B/'OFFICIAL_NORMAL_v1')], D, 'STEP2_gate_slow')
            lines = (B/'OFFICIAL_SLOW_v1/judge.txt').read_text() if (B/'OFFICIAL_SLOW_v1/judge.txt').exists() else rs.stdout + rs.stderr
            log(D, 'STEP2 slow gate rc=%d\n%s' % (rs.returncode, '\n'.join(l for l in lines.splitlines() if l.startswith(('  ok','  FAIL')))))
            if rs.returncode != 0:
                gate_fail.append('slow')
                if not a.ignore_gate_fail: raise SystemExit('STOP: slow early gate failed')
    # 3 collectors retirement
    ids = [ex['normal_guard_pid'], ex['slow_guard_pid']] + [read(B/m/'guard.json')['pid'] for m in ('normal','slow')]
    for m in ('normal','slow'):
        g = read(B/m/'guard.json'); assert g['completed'] and g['exit_code'] == 0 and g['whole_group_retired'] and g['remaining_rows'] == []
        l = read(B/m/'launcher.json'); assert l['client_exit'] == 0 and l['server_exit'] == 0 and l['sha256'] == h
    assert absent(ids); put(D/'COLLECTORS_RETIREMENT_v1.json', dict(utc=utc(), owned_ids=ids, all_owned_PID_PPID_PGID_absent=True, exits=ex))
    log(D, 'STEP3 collectors retired')
    # 4 stage A
    r = run([PY, '-B', str(P/'stage_a_generic_v1.py'), str(src), f'a_{tag}'] + (['--reviewed', a.reviewed] if a.reviewed else []), D, 'STEP4_stage_a', cwd=O)
    A = RUNS/f'a_{tag}'; st = read(A/'stage_a.json'); arc = A/'submission.tar.gz'
    log(D, f"STEP4 stage A passed={st['passed']} {st['line']}")
    assert st['passed'] and st['candidate_sha256'] == h
    with tarfile.open(arc) as f: assert sorted(f.getnames()) == ['estimator.py', 'manifest.json'] and f.extractfile('estimator.py').read() == src.read_bytes()
    # 5 official120 finalization (sequence of the b709 release with source/tag substituted)
    steps = [[PY, '-B', str(REL/'fastgate_pin_v1.py'), '--check', str(REL/'fastgate_pin_v1.json'), str(src), str(B/'normal'), str(B/'slow')],
             ('copy', B/'normal', RUNS/f'fg_{tag}_official120_n_src/C0'),
             [PY, '-B', str(OFF/'stage_c_official120_v1.py'), str(src), f'c_{tag}_official120', '--from-run', str(RUNS/f'fg_{tag}_official120_n_src'), '--short-setup', '--legs', 'C', '--parent-from', str(RUNS/'c_S42_bank_legs_heavy')],
             ('copy', B/'slow', RUNS/f'fg_{tag}_official120_s_src/C0'),
             [PY, '-B', str(OFF/'stage_c_official120_v1.py'), str(src), f'{tag}_official120_slow', '--from-run', str(RUNS/f'fg_{tag}_official120_s_src'), '--setup-slowdown', '1.3', '--legs', 'C', '--parent-from', str(RUNS/'c_S42_bank_legs_heavy')],
             [PY, '-B', str(OFF/'designation_judge_official120_v1.py'), f'{tag}_official120', '--pair', str(RUNS/f'c_{tag}_official120'), '--slow', str(RUNS/f'{tag}_official120_slow'), '--worker', str(RUNS/'none'), '--first-priced' if a.safe else '--primary'],
             [PY, '-B', str(OFF/'release_gate_official120_v1.py'), f'{tag}_official120', '--stage-a', str(A), '--pair', str(RUNS/f'c_{tag}_official120'), '--slow', str(RUNS/f'{tag}_official120_slow'), '--judge', str(RUNS/f'judge_{tag}_official120.txt'), '--allow-missing-worker', '--first-priced' if a.safe else '--primary']]
    gate = None
    for i, s in enumerate(steps):
        if isinstance(s, tuple):
            _, s0, s1 = s; s1.parent.mkdir(parents=True); shutil.copytree(s0, s1)
            assert all(sha(x) == sha(s1/x.relative_to(s0)) for x in s0.rglob('*') if x.is_file()); continue
        rr = run(s, D, f'STEP5_{i:02d}')
        if Path(s[2]).name == 'designation_judge_official120_v1.py':
            put(RUNS/f'judge_{tag}_official120.txt', rr.stdout)
            bad = '  FAIL ' in rr.stdout
            log(D, 'STEP5 designation judge\n' + '\n'.join(l for l in rr.stdout.splitlines() if l.startswith(('  ok','  FAIL'))))
            if bad: gate_fail.append('designation')
            if bad and not a.ignore_gate_fail: raise SystemExit('STOP: designation judge FAIL')
            continue
        if Path(s[2]).name == 'release_gate_official120_v1.py':
            gate = json.loads(rr.stdout) if rr.stdout.strip().startswith('{') else dict(passed=False, raw=rr.stdout[-2000:])
            put(D/'RELEASE_GATE_OFFICIAL120_v1.json', gate); log(D, f"STEP5 release gate passed={gate.get('passed')}")
            if not gate.get('passed'):
                gate_fail.append('release_gate')
                if not a.ignore_gate_fail: raise SystemExit('STOP: release gate FAIL')
            continue
        assert rr.returncode == 0, (s, rr.stdout[-1500:], rr.stderr[-1500:])
    # 6 captured-setup native parity
    N = D/'parity'; N.mkdir()
    for m in ('normal','slow'):
        man = B/m/'manifest.json'
        rr = run(['nice', '-n', '10', NPY, '-B', str(P/'native_probe_generic_v1.py'), str(src), '35,100', str(N/f'tiny_{m}')], D, f'STEP6_tiny_{m}',
                 e=env(SETUP_ONLY='1', CAPTURED_SETUP_MANIFEST=str(man), CANDIDATE_SHA256=h))
        assert rr.returncode == 0 and 'CAPTURED_SETUP_REPLAY_PASS' in rr.stdout, rr.stdout[-1500:] + rr.stderr[-1500:]
    log(D, 'STEP6 setup-only replays PASS (all semantic setup phases exact)')
    procs = {}
    for m in ('normal','slow'):
        admit(D, 1)
        cmd = [PY, '-B', str(REL/'guard_footprint_v3.py'), '--record', str(N/f'guard_{m}.json'), '--seconds', '1100', '--footprint-limit', '7500000000',
               '--rss-backstop', '10000000000', 'nice', '-n', '10', NPY, '-B', str(P/'native_probe_generic_v1.py'), str(src), '35,100', str(N/m)]
        procs[m] = subprocess.Popen(cmd, cwd=REL, env=env(CAPTURED_SETUP_MANIFEST=str(B/m/'manifest.json'), CANDIDATE_SHA256=h),
                                    stdout=open(N/f'{m}.log', 'x'), stderr=subprocess.STDOUT); procs[m].wait()   # sequential (memory)
    rc = {m: p.wait() for m, p in procs.items()}; assert rc == {'normal': 0, 'slow': 0}, rc
    rows, pids = [], [p.pid for p in procs.values()]
    for m in ('normal','slow'):
        g = read(N/f'guard_{m}.json'); pids.append(g['pid']); assert g['completed'] and g['exit_code'] == 0 and g['whole_group_retired']
        nm = read(N/m/'manifest.json'); assert nm['sha256'] == h and nm['setup']['captured_replay']['all_setup_counts_exact']
        a_ = [json.loads(x) for x in (N/m/'rows.jsonl').read_text().splitlines()]; b_ = [json.loads(x) for x in (B/m/'rows.jsonl').read_text().splitlines()]
        assert [x['rank'] for x in a_] == [x['rank'] for x in b_] == [35, 100]
        for x, y in zip(a_, b_):
            k = x['rank']; nat = np.load(N/m/f'pred_{k}.npy'); prod = np.frombuffer((B/m/f'pred_{k}.bin').read_bytes(), dtype=np.float32).reshape(16, 1024).astype(np.float64)
            rows.append(dict(mode=m, rank=k, bytes_equal=bool(nat.tobytes() == prod.tobytes()), native_flops=x['flops'], production_flops=y['flops']))
    assert absent(pids)
    par = dict(passed=all(r_['bytes_equal'] and r_['native_flops'] == r_['production_flops'] for r_ in rows), candidate_sha256=h, rows=rows, owned_ids=pids, all_retired=True)
    put(D/'CAPTURED_SETUP_PARITY_v1.json', par); log(D, f"STEP6 parity passed={par['passed']} {rows}")
    assert par['passed']
    # 7 confirmation
    ev = [src, arc, A/'stage_a.json', D/'COLLECTORS_RETIREMENT_v1.json', D/'CAPTURED_SETUP_PARITY_v1.json', D/'RELEASE_GATE_OFFICIAL120_v1.json', POLICY] + science
    ev += [B/x/'early_gate.json' for x in ('OFFICIAL_NORMAL_v1', 'OFFICIAL_SLOW_v1') if (B/x/'early_gate.json').exists()]
    argv = [PY, '-B', str(O/'fire/fire_once.py'), tag, str(arc), a.desc]
    status = 'DRY_RUN_NOT_FOR_FIRE' if (a.dry_run or gate_fail) else 'CONFIRMED'
    put(D/'RELEASE_CONFIRMED_v1.json', dict(status=status, utc=utc(), executed_by='opus-release', candidate_path=str(src), candidate_sha256=h, archive_path=str(arc),
        archive_sha256=sha(arc), gate_failures=gate_fail, official_wall_policy=dict(path=str(POLICY), sha256=sha(POLICY)), fire_argv_for_lead_only=argv if status == 'CONFIRMED' else None,
        fire_cwd=str(O), submitted=False, evidence_sha256={str(x): sha(x) for x in ev}))
    log(D, f'STEP7 {status} confirmation {sha(D/"RELEASE_CONFIRMED_v1.json")[:12]} archive {sha(arc)[:12]}')
    # 8 prefire
    rr = run([PY, '-B', str(O/'fire/prefire_check_v1.py'), str(arc), h[:8], str(a.ref_archive)], D, 'STEP8_prefire', cwd=O)
    tail = [l for l in rr.stdout.splitlines() if not l.startswith('  eye')][-3:]
    log(D, f'STEP8 prefire rc={rr.returncode} ' + ' | '.join(tail))
    clear = rr.returncode == 0 and 'PREFIRE CLEAR | new signatures: 0' in rr.stdout
    put(D/'PIPELINE_DONE_v1.json', dict(utc=utc(), status=status, prefire_clear_zero_new=clear, gate_failures=gate_fail,
        fire_command=(' '.join(argv[:-1]) + f" '{a.desc}'") if (status == 'CONFIRMED' and clear) else None))
    log(D, f"DONE status={status} prefire_clear={clear}" + (f"\nFIRE (lead only, cwd O): {' '.join(argv[:-1])} '{a.desc}'" if status == 'CONFIRMED' and clear else ''))

if __name__ == '__main__':
    main()
