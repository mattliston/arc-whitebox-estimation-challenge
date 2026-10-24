"""Metadata-only early primary count gate using pinned release calculations.

No sessions, source builds, release triggers or submissions are launched.
Usage: candidate.py session_dir normal|slow fresh_assessment_dir [normal_assessment_dir]
Writes an immutable stage-C assessment and invokes the existing numeric judge.
The full later release gate remains mandatory.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

D = Path(__file__).resolve().parent
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
R = O/'release'
sys.path.insert(0, str(R))
import common
import project
import stage_c_boundary as stage
# Root official policy changes only the projected hard wall; 100 seconds remains advisory.
POLICY=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json')
assert hashlib.sha256(POLICY.read_bytes()).hexdigest()=='f7ed781f660a20d2687b8d39289258e41338ccaf38a1bcd8d71c7c513f4ba132'
assert json.loads(POLICY.read_text())["approved"] is True
common.LIMITS=dict(common.LIMITS,worst_wall_s=120.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def assess(candidate, session, output, slow, normal=None):
    candidate, session, output = map(lambda p:Path(p).resolve(), (candidate,session,output))
    assert not output.exists()
    manifest = json.loads((session/'manifest.json').read_text())
    launcher = json.loads((session/'launcher.json').read_text())
    assert manifest['sha256'] == launcher['sha256'] == sha(candidate)
    assert manifest['ranks'] == [35,100]
    assert manifest['short_setup'] is True and manifest['setup_slowdown'] == (1.3 if slow else 1.0)
    assert manifest['sites'] is False and manifest['profiled'] is False
    pin = json.loads((R/'fastgate_pin_v1.json').read_text())
    assert manifest['client_flopscope'] == pin['client_flopscope']
    assert launcher['client_exit'] == launcher['server_exit'] == 0
    csession = stage.load_session(session)
    guard = csession['guard']
    assert guard['completed'] is True and guard['exit_code'] == 0 and guard.get('error') is None
    assert guard['whole_group_retired'] is True and guard['remaining_rows'] == []
    assert [row['rank'] for row in csession['rows']] == [35,100]
    bankdir = R/'runs/c_S42_bank_legs_heavy'
    pdirs = sorted(p for p in bankdir.iterdir() if p.is_dir() and p.name.startswith('P'))
    assert [p.name for p in pdirs] == ['P0']
    psessions = []
    for path in pdirs:
        m = json.loads((path/'manifest.json').read_text())
        assert m['sha256'] == sha(common.BANK) and m['ranks'] == [35,100]
        psessions.append(stage.load_session(path))
    sessions = {'C':[csession], 'P':psessions}
    clean = lambda arm:{r['rank'] for s in sessions[arm] for r in s['rows'] if project.is_clean(r)}
    covered = all({35,100} <= clean(arm) for arm in ('C','P'))
    keep = project.is_clean if covered else None
    c, p = project.summarize(sessions['C'],keep), project.summarize(sessions['P'],keep)
    projection = project.project(c,p,common.PLATFORM)
    guards = [s['guard'] for arm in sessions.values() for s in arm]
    footprint = common.peak_footprint([s for arm in sessions.values() for s in arm])
    swap = max(g['max_swap_growth_bytes'] for g in guards)
    gates = dict(memory=footprint < common.LIMITS['footprint_bytes'],
                 swap=swap <= common.LIMITS['swap_bytes'], nonfinite=c['nonfinite']==0,
                 clean=covered, **stage.setup_gates(sessions['C'],c),
                 residual=projection['residual_max_s'] <= common.LIMITS['residual_s'],
                 wall=projection['worst_wall_s'] <= common.LIMITS['worst_wall_s'])
    result = dict(stage='C', candidate=str(candidate), candidate_sha256=sha(candidate),
                  parent=str(common.BANK), parent_sha256=sha(common.BANK),
                  ranks='35,100', legs='C', candidate_summary=c, parent_summary=p,
                  projection=projection, max_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
                  max_footprint_bytes=footprint,max_swap_growth_bytes=swap,limits=common.LIMITS,
                  sessions={arm:[s['dir'] for s in group] for arm,group in sessions.items()},
                  gates=gates, stress_slowdown=1.3 if slow else None,
                  official_wall_policy=str(POLICY), official_wall_policy_sha256=sha(POLICY), advisory_wall_s=100.0, advisory_wall_pass=projection['worst_wall_s']<=100.0, scope='Versioned official 120-second hard wall assessment; 100-second advisory preserved; every other gate unchanged')
    output.mkdir()
    exclusive(output/'stage_c.json',result)
    pair = Path(normal).resolve() if slow else output
    slowdir = output if slow else output/'_missing_slow'
    assert not (output/'_missing_worker').exists()
    command = [sys.executable,'-B','/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/designation_judge_official120_v1.py','s56gi1_early_'+('slow' if slow else 'normal'),
               '--pair',str(pair),'--slow',str(slowdir),'--worker',str(output/'_missing_worker'),'--primary']
    judged = subprocess.run(command,cwd=R,capture_output=True,text=True,check=True,timeout=120)
    with (output/'judge.txt').open('x') as stream:
        stream.write(judged.stdout)
    expected_missing = 'no result for: worker' if slow else 'no result for: slow setup, worker'
    lines = judged.stdout.splitlines()
    assert lines and expected_missing in lines[-1] and '] INCOMPLETE:' in lines[-1], judged.stdout
    expected_names = ['P(loss) per 100 networks (primary / twin)',
                      'binding phase at the platform pace',
                      'projected platform setup wall (cap 5 s, whole run)',
                      'projected worst-network wall']
    if slow:
        expected_names += ['requests under a setup 1.3x slower',
                           'projected setup wall under a 1.3x slow setup']
    criteria = [line for line in lines if line.startswith(('  ok  ','  FAIL '))]
    assert len(criteria) == len(expected_names)
    assert all(any(name in line for line in criteria) for name in expected_names)
    failed = [line.strip() for line in criteria if line.startswith('  FAIL ')]
    stage_failed = [name for name,value in gates.items() if not value and name not in ('swap','clean')]
    evidence = [candidate,session/'manifest.json',session/'launcher.json',session/'guard.json',session/'rows.jsonl',
                output/'stage_c.json',output/'judge.txt',POLICY,Path(__file__),Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/designation_judge_official120_v1.py')]
    if slow:
        normal_gate = json.loads((pair/'early_gate.json').read_text())
        assert normal_gate['passed'] is True
        evidence.extend([pair/'stage_c.json',pair/'early_gate.json'])
    receipt = dict(utc_from_date_u=subprocess.check_output(['/bin/date','-u','+%FT%TZ'],text=True).strip(),
                   passed=not failed and not stage_failed,mode='slow' if slow else 'normal',
                   scope='Early numerical count limits only; full stage-A/release/native/seal/prefire still required',
                   numeric_judge_command=command,judged_criteria=criteria,failed_criteria=failed,
                   failed_stage_c_gates=stage_failed,
                   warnings=[name for name in ('swap','clean') if not gates[name]],
                   evidence_sha256={str(path):sha(path) for path in evidence})
    exclusive(output/'early_gate.json',receipt)
    return receipt


if __name__ == '__main__':
    assert len(sys.argv) in (5,6)
    candidate,session,mode,out = sys.argv[1:5]
    assert mode in ('normal','slow') and (len(sys.argv)==6)==(mode=='slow')
    receipt = assess(candidate,session,out,mode=='slow',sys.argv[5] if mode=='slow' else None)
    print(json.dumps(receipt),flush=True)
    raise SystemExit(0 if receipt['passed'] else 1)
