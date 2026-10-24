"""[opus-release] v3 FINISH of a real-file release (release_gov_realfile_v2.py) that stopped after its STEPC: re-verifies STEPA-C on disk
(the governor-threshold-only diff, stage A of the real file with the archive holding its exact bytes, the captured-setup parity record) and
then runs STEPD-F with the provenance made SAFE-aware: a parity release run with --safe has no primary early gates; its first-priced
designation judge lines (RUNS/judge_<paritytag>_official120.txt) and its release gate stand in their place. Writes RELEASE_CONFIRMED and
PIPELINE_DONE into the SAME rel dir (only files that do not exist yet). NEVER fires.
usage: release_gov_realfile_finish_v3.py --rel REL_DIR --parity-rel PARITY_REL_DIR --s2tr-boundary DIR [--science @list]"""
import argparse, importlib.util, json, re, tarfile
from pathlib import Path

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rp5', P/'release_pipeline_v5.py'); rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
O, RUNS, PY = rp.O, rp.RUNS, rp.PY
sha, read, put, utc, log, run = rp.sha, rp.read, rp.put, rp.utc, rp.log, rp.run
PAT = re.compile(r'^(_GOV_[A-Z0-9_]*(?:CAP|WALL)[A-Z0-9_]*_S) = (.+?)(\s*#.*)?$')


def rows_of(d):
    return [json.loads(x) for x in (d/'rows.jsonl').read_text().splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rel', required=True, type=Path); ap.add_argument('--parity-rel', required=True, type=Path)
    ap.add_argument('--s2tr-boundary', required=True, type=Path); ap.add_argument('--science', default=''); ap.add_argument('--ref-archive', type=Path, default=rp.REF)
    a = ap.parse_args(); D, R = a.rel.resolve(), a.parity_rel.resolve(); B = R/'boundary'
    st = read(D/'START_v1.json'); src = Path(st['source']); h = st['source_sha256']; tag = st['tag']; desc = st['desc']
    assert sha(src) == h and sha(Path(st['parity_source'])) == st['parity_source_sha256']
    for name in ('NUMERICS_PROVENANCE_v1.json', 'RELEASE_CONFIRMED_v1.json', 'PIPELINE_DONE_v1.json'):
        assert not (D/name).exists(), f'{name} already exists: nothing to finish'
    pst = read(R/'START_v1.json'); assert pst['source_sha256'] == st['parity_source_sha256'], 'parity rel is not this parity copy'
    # re-verify A: threshold-only diff
    a_l, b_l = Path(st['parity_source']).read_text().splitlines(), src.read_text().splitlines(); assert len(a_l) == len(b_l)
    diff = []
    for x, y in zip(a_l, b_l):
        if x != y:
            mx, my = PAT.match(x), PAT.match(y)
            assert mx and my and mx.group(1) == my.group(1) and float(mx.group(2)) == 1e9, (x, y)
            diff.append((mx.group(1), mx.group(2), my.group(2)))
    assert diff
    # re-verify B: stage A of the real file
    A = RUNS/f'a_{tag}'; sta = read(A/'stage_a.json'); arc = A/'submission.tar.gz'
    assert sta['passed'] and sta['candidate_sha256'] == h
    with tarfile.open(arc) as f:
        assert sorted(f.getnames()) == ['estimator.py', 'manifest.json'] and f.extractfile('estimator.py').read() == src.read_bytes()
    # re-verify C: captured-setup parity record
    par = read(D/'CAPTURED_SETUP_PARITY_v1.json'); assert par['passed'] and par['candidate_sha256'] == h and par['production_source_sha256'] == st['parity_source_sha256']
    log(D, f'FINISH v3: STEPA-C re-verified on disk (diff {diff}; stage A {sha(arc)[:12]}; parity passed)')
    # D provenance (SAFE-aware)
    gate = read(R/'RELEASE_GATE_OFFICIAL120_v1.json')
    early = {x: read(B/x/'early_gate.json') for x in ('OFFICIAL_NORMAL_v1', 'OFFICIAL_SLOW_v1') if (B/x/'early_gate.json').exists()}
    judge_txt = RUNS/f"judge_{pst['tag']}_official120.txt"
    judge_lines = [l for l in judge_txt.read_text().splitlines() if l.startswith(('  ok', '  FAIL'))] if judge_txt.exists() else []
    safe_mode = not early
    S = a.s2tr_boundary.resolve(); ident = {}
    for m in ('normal', 'slow'):
        for x, y in zip(rows_of(B/m), rows_of(S/m)):
            k = x['rank']; assert k == y['rank']
            ident[f'{m}_{k}'] = dict(pred_bytes_equal=(B/m/f'pred_{k}.bin').read_bytes() == (S/m/f'pred_{k}.bin').read_bytes(),
                                     **{f: [x.get(f), y.get(f)] for f in ('requests', 'flops', 'free_trips') if f in x or f in y})
    prov = dict(official120_from=f'parity copy {Path(st["parity_source"]).name} {st["parity_source_sha256"][:8]} (differs from the real file only in the governor thresholds {diff}); boundary / {"first-priced SAFE designation judge" if safe_mode else "early gates + designation judge"} / release gate ran in {R}',
                safe_mode=safe_mode, parity_release_gate_passed=bool(gate.get('passed')),
                early_gates={k: v.get('passed', v.get('status')) for k, v in early.items()}, designation_judge_lines=judge_lines,
                s2tr_boundary=str(S), parity_boundary_vs_s2tr=ident,
                statement='only the parity copy\'s own official120 numbers apply (the stack changes counts vs S2tr); the real file equals the parity copy natively (STEPC) and on the untriggered production path by construction (threshold-only diff)')
    assert prov['parity_release_gate_passed'] and judge_lines and not any(l.startswith('  FAIL') for l in judge_lines)
    put(D/'NUMERICS_PROVENANCE_v1.json', prov); log(D, f"STEPD provenance (SAFE-aware={safe_mode}): parity release gate passed; judge lines {len(judge_lines)} all ok")
    # E confirmation
    items = Path(a.science[1:]).read_text().split() if a.science.startswith('@') else [x for x in a.science.split(',') if x]
    science = [Path(x).resolve() for x in items]; assert all(x.is_file() for x in science)
    ev = [src, Path(st['parity_source']), arc, A/'stage_a.json', D/'CAPTURED_SETUP_PARITY_v1.json', D/'NUMERICS_PROVENANCE_v1.json',
          R/'RELEASE_GATE_OFFICIAL120_v1.json', R/'RELEASE_CONFIRMED_v1.json', R/'COLLECTORS_RETIREMENT_v1.json', rp.POLICY] + science
    ev += [B/x/'early_gate.json' for x in early] + ([judge_txt] if judge_txt.exists() else [])
    argv = [PY, '-B', str(O/'fire/fire_once.py'), tag, str(arc), desc]
    put(D/'RELEASE_CONFIRMED_v1.json', dict(status='CONFIRMED', utc=utc(), executed_by='opus-release (finish v3)', candidate_path=str(src), candidate_sha256=h,
        archive_path=str(arc), archive_sha256=sha(arc), numerics_provenance=prov, official_wall_policy=dict(path=str(rp.POLICY), sha256=sha(rp.POLICY)),
        fire_argv_for_lead_only=argv, fire_cwd=str(O), submitted=False, evidence_sha256={str(x): sha(x) for x in ev}))
    log(D, f'STEPE CONFIRMED confirmation {sha(D/"RELEASE_CONFIRMED_v1.json")[:12]} archive {sha(arc)[:12]}')
    # F prefire
    rr = run([PY, '-B', str(O/'fire/prefire_check_v1.py'), str(arc), h[:8], str(a.ref_archive)], D, 'STEPF_prefire', cwd=O)
    tail = [l for l in rr.stdout.splitlines() if not l.startswith('  eye')][-3:]
    log(D, f'STEPF prefire rc={rr.returncode} ' + ' | '.join(tail))
    clear = rr.returncode == 0 and 'PREFIRE CLEAR | new signatures: 0' in rr.stdout
    put(D/'PIPELINE_DONE_v1.json', dict(utc=utc(), status='CONFIRMED', prefire_clear_zero_new=clear,
        fire_command=(' '.join(argv[:-1]) + f" '{desc}'") if clear else None))
    log(D, f'DONE status=CONFIRMED prefire_clear={clear}' + (f"\nFIRE (lead only, cwd O): {' '.join(argv[:-1])} '{desc}'" if clear else ''))


if __name__ == '__main__':
    main()
