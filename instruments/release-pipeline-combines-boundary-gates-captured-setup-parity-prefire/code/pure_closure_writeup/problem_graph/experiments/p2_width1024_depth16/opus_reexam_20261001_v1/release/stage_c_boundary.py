"""TAG opus-release STAGE C: boundary census of a candidate, PAIRED with a parent of known platform telemetry (default: the bank
S35 = #333111), on the production path (numpy-free client -> flopscope server, OpenBLAS, 1 thread, setup() first).

Worker sessions run one at a time in the order given by --legs (default PCCP: parent, candidate, candidate, parent, so slow
drift of the machine cancels). Each session = setup() + the networks of --ranks (default 35,100; first one is cold).
Reported per arm (mean over sessions and networks): requests, hidden free round trips, kernel seconds, kernel cycles and
instructions (server_light.py), overhead, residual, setup time, handles, peak footprint (server + client) and group RSS.
Projection onto the platform (project.py): the bank's measured platform numbers scaled by the candidate/parent ratios.
One PASS / PASS* / FAIL line (common.judge); limits as used for S35: server + client peak footprint < 7.5 GB (group RSS is only the
guard's backstop at 10 GB since 2026-10-02 08:50Z), zero added swap, setup <= 4.1 s, projected residual max <= 0.4 s,
projected worst-network wall <= 105 s. (Handles are gated in stage D, over 20 networks.)
--short-setup runs every session's setup under the platform's clock instead of this machine's (setup_clock.py): how far the
setup gets before its guard is then the same in every session and that of a normal platform worker; --setup-slowdown X is the
same for a worker X times slower in setup. In those modes the wall-clock setup gate is replaced by `binding` (every tree key bound
before the guard, as on the bank's normal platform workers) and the setup must not have fallen back (`setup_ok`).
--platform-record FILE: the parent's own platform telemetry (the stored evaluation JSON of its graded submission) when the
parent is not S35: the projection then scales THAT record, e.g. --parent <S38 file> --platform-record
O/fire/s38_20261002/eval_333481.json (the nearest graded file is the best parent: less is extrapolated).
usage: stage_c_boundary.py <candidate.py> <tag> [--parent PATH [--platform-record FILE]] [--ranks 35,100] [--legs PCCP] [--no-gate]
       [--from-run DIR] [--short-setup | --setup-slowdown X]
       [--legs PP]           (parent legs only: stores the bank's legs for later --parent-from runs; the candidate is not run)
       [--parent-from DIR]   (stored parent legs: instruction and request counts repeat to +-0.5% between clean sessions)
       -> O/release/runs/<tag>/{P1,C1,...}/rows.jsonl, runs/<tag>/stage_c.json"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common import BANK, BANK_NAME, LIGHT_PY, LIMITS, PLATFORM, R, guarded, judge, peak_footprint, run_dir, sha, write_json
import project


def busy():
    """Other processes using more than half a core right now, as (pid, %CPU, command): a session measured beside them was not
    measured on a quiet machine (two lanes ran into the quiet window of 2026-10-02 07:25Z and only their own reports showed it)."""
    listing = subprocess.run(['ps', '-Ao', 'pid=,pcpu=,command='], capture_output=True, text=True).stdout
    rows = [line.split(None, 2) for line in listing.splitlines()]
    return [(int(r[0]), float(r[1]), r[2][:400]) for r in rows if len(r) == 3 and float(r[1]) > 50 and int(r[0]) != os.getpid()]


def session(out, candidate, ranks, seconds, gate, extra=()):
    out.mkdir()
    if gate:
        subprocess.run(['bash', str(R/'own_gate.sh'), '7', 'release'], check=True)
    beside = busy()
    (out/'busy_at_start.json').write_text(json.dumps(beside) + '\n')
    if beside:
        script = lambda cmd: next((x for x in cmd.split() if x.endswith('.py')), cmd.split()[0]).rsplit('/', 1)[-1]
        print(f"  {out.name}: {len(beside)} other busy processes at its start: "
              + '; '.join(f'{script(cmd)} {c:.0f}% (pid {pid})' for pid, c, cmd in beside[:6]), flush=True)
    g = guarded(out/'guard.json', seconds, [LIGHT_PY, '-B', R/'boundary_run.py', candidate, ranks, out, *extra], out/'guard.log')
    if not g['completed']:
        raise SystemExit(f"session {out.name} did not complete: {g['error']} (see {out}/guard.log)")
    return load_session(out)


def load_session(out):
    """A finished worker session directory as a dict (rows, setup, guard record, server receipt)."""
    out = Path(out)
    rows = [json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
    manifest = json.loads((out/'manifest.json').read_text())
    launcher = json.loads((out/'launcher.json').read_text())
    return dict(dir=str(out), rows=rows, setup=manifest['setup'], guard=json.loads((out/'guard.json').read_text()),
                server=launcher['server'])


def setup_text(setup):
    """What a session's setup got done: tree keys bound of the list, deferred keys fused, requests, and which clock ran it."""
    log = {x[0]: x[1] for x in setup['pb_log']}
    total = setup.get('keys_total')
    bound = f"{log.get('wall guard', total if total is not None else 'all')}" + (f" of {total}" if total is not None else '')
    clock = ('own wall clock' if setup.get('slowdown') is None else 'no guard' if setup['slowdown'] == 0
             else f"platform clock x{setup['slowdown']:g}")
    return f"{bound} keys bound, {log.get('lvfuse', 0)} deferred keys fused, {setup['requests']} requests, {clock}"


BINDING_SHARE = 0.90


def setup_gates(sessions, summary):
    """Under the candidate's own wall clock: the setup wall (load-dependent). Under the platform's clock at a normal worker's
    pace: at least BINDING_SHARE of the listed keys bound before the guard in every session (until 2026-10-10 02:22Z: every
    key; since files carry the constants lever, the platform's own setups stop short of the list's end, #333607: 621-632 of
    649, and what the unbound keys cost is in the networks' request counts). At any other pace (a stress run) how far the
    setup gets is the input, not a gate. Always: the setup did not fall back ('restored after failure' in its log)."""
    ok = not any(x[0] == 'restored after failure' for s in sessions for x in s['setup']['pb_log'])
    paces = {s['setup'].get('slowdown') for s in sessions}
    if paces == {None}:
        return dict(setup=summary['setup_s_max'] <= LIMITS['setup_s'], setup_ok=ok)
    if paces == {1.0}:
        def bound(s):
            log = {x[0]: x[1] for x in s['setup']['pb_log'] if len(x) > 1}
            total = s['setup'].get('keys_total') or 649
            return log.get('wall guard', total)/total
        return dict(binding=all(bound(s) >= BINDING_SHARE for s in sessions), setup_ok=ok)
    return dict(setup_ok=ok)


def stress(sessions):
    """The setup slowdown of a stress run (candidate sessions set up at the pace of a slower worker), else None."""
    paces = {s['setup'].get('slowdown') for s in sessions}
    return paces.pop() if len(paces) == 1 and paces not in ({None}, {0.0}, {1.0}) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate', type=Path)
    ap.add_argument('tag')
    ap.add_argument('--parent', type=Path, default=BANK)
    ap.add_argument('--ranks', default='35,100')
    ap.add_argument('--legs', default='PCCP')
    ap.add_argument('--no-gate', action='store_true')
    ap.add_argument('--short-setup', action='store_true', help="every session's setup runs under the platform's clock "
                    '(setup_clock.py): a normal platform worker')
    ap.add_argument('--setup-slowdown', type=float, help='the same for a worker this many times slower in setup (0 = no guard)')
    ap.add_argument('--sites', action='store_true', help="record every request's call site and operation (request_sites.py); "
                    'a diagnostic run: the residual counters are inflated')
    ap.add_argument('--profile', action='store_true', help='cProfile every prediction in the client (residual_report.py); the '
                    "run's counters are then inflated: not for a verdict")
    ap.add_argument('--from-run', type=Path, help='re-assess the finished sessions of an earlier stage-C run directory')
    ap.add_argument('--skip', default='', help='with --from-run: comma-separated leg positions to leave out (e.g. 2 for the third leg)')
    ap.add_argument('--parent-from', type=Path, help='take the parent arm from the P legs of an earlier stage-C run directory '
                    '(same parent source, same ranks) and run only candidate legs: use --legs C or CC with it')
    ap.add_argument('--platform-record', type=Path, help="the parent's stored platform evaluation JSON, when the parent is not S35")
    a = ap.parse_args()
    candidate, parent = a.candidate.resolve(), a.parent.resolve()
    assert a.platform_record or parent == BANK.resolve() or 'C' not in a.legs, 'a parent other than the bank needs its own --platform-record'
    platform = project.platform_record(a.platform_record) if a.platform_record else PLATFORM
    parent_name = (BANK_NAME if parent == BANK.resolve() else parent.parent.name if parent.name == 'candidate.py'
                   else parent.name[:40]) + f" #{platform['sid']}"
    out = run_dir(a.tag)
    load = os.getloadavg()[0]
    n = len(a.ranks.split(','))
    slowdown = a.setup_slowdown if a.setup_slowdown is not None else 1.0 if a.short_setup else None
    setup_args = (([] if slowdown is None else ['--setup-slowdown', str(slowdown)]) + (['--profile'] if a.profile else [])
                  + (['--sites'] if a.sites else []))
    sessions = {'P': [], 'C': []}
    skip = {int(x) for x in a.skip.split(',') if x}
    for i, leg in enumerate(a.legs):
        if i in skip:
            continue
        source = parent if leg == 'P' else candidate
        if a.from_run:
            s = load_session(a.from_run/f'{leg}{i}')
            assert json.loads((a.from_run/f'{leg}{i}'/'manifest.json').read_text())['sha256'] == sha(source), 'leg ran another source'
        else:
            s = session(out/f'{leg}{i}', source, a.ranks, 600 + 600*n, not a.no_gate, setup_args)   # a hang guard, not a speed limit
        sessions[leg].append(s)
        r = project.summarize([s])
        print(f"  leg {leg}{i}: kernel {r['kernel_s']:.2f} s, {r['kernel_cycles']/1e9:.1f} Gcycles, "
              f"{r['kernel_instructions']/1e9:.1f} Ginstr ({r['kernel_instructions_fault_free']/1e9:.1f} without the work of "
              f"{r['minor_faults']/1e3:.0f}k page faults); requests {r['requests']:.0f}; residual {r['residual_s']:.3f} s; "
              f"setup {r['setup_s']:.2f} s, {setup_text(s['setup'])}; server system/user {r['system_share']:.2f}", flush=True)
    if a.parent_from:
        assert 'P' not in a.legs, 'with --parent-from run candidate legs only'
        for leg in sorted(x for x in a.parent_from.iterdir() if x.is_dir() and x.name.startswith('P')):
            assert json.loads((leg/'manifest.json').read_text())['sha256'] == sha(parent), 'stored parent legs ran another source'
            assert json.loads((leg/'manifest.json').read_text())['ranks'] == [int(x) for x in a.ranks.split(',')]
            sessions['P'].append(load_session(leg))
    if not sessions['C']:      # --legs P or PP: the parent's own legs, stored for later --parent-from runs (quick.sh)
        s = project.summarize(sessions['P'])
        line = (f"[C boundary] BANK LEGS: {len(sessions['P'])} session(s) of {parent.parent.name}: requests/network "
                f"{s['requests']:.0f}, kernel instructions {s['kernel_instructions']/1e9:.1f} G, {setup_text(sessions['P'][-1]['setup'])}; "
                f"paged network measurements: {s['paged_ranks'] or 'none'}; calm for the residual counter: {s['calm']}")
        print(line, flush=True)
        write_json(out/'stage_c.json', dict(stage='C', parent=str(parent), parent_sha256=sha(parent), ranks=a.ranks, legs=a.legs,
                                            parent_summary=s, sessions={'P': [x['dir'] for x in sessions['P']]}, line=line))
        sys.exit(0)
    # Networks during which the server paged read high and are left out of BOTH arms' counters; the gate `clean` asks that every
    # network still has a clean measurement in each arm.
    ranks = [int(x) for x in a.ranks.split(',')]
    clean = lambda arm: {r['rank'] for s in sessions[arm] for r in s['rows'] if project.is_clean(r)}
    covered = all(set(ranks) <= clean(arm) for arm in ('C', 'P') if sessions[arm])
    keep = project.is_clean if covered else None
    c, p = project.summarize(sessions['C'], keep), project.summarize(sessions['P'], keep) if sessions['P'] else None
    proj = project.project(c, p, platform) if p else None
    guards = [s['guard'] for leg in sessions.values() for s in leg]
    rss = max(g['max_rss_bytes'] for g in guards)
    footprint = peak_footprint([s for leg in sessions.values() for s in leg])
    swap = max(g['max_swap_growth_bytes'] for g in guards)
    result = dict(stage='C', candidate=str(candidate), candidate_sha256=sha(candidate), parent=str(parent),
                  parent_sha256=sha(parent), ranks=a.ranks, legs=a.legs, candidate_summary=c, parent_summary=p, projection=proj,
                  max_group_rss_bytes=rss, max_footprint_bytes=footprint, max_swap_growth_bytes=swap, limits=LIMITS,
                  sessions={k: [s['dir'] for s in v] for k, v in sessions.items()})
    paged = sorted(f"{arm}{r['rank']}" for arm in ('P', 'C') for s in sessions[arm] for r in s['rows'] if not project.is_clean(r))
    gates = dict(memory=footprint < LIMITS['footprint_bytes'], swap=swap <= LIMITS['swap_bytes'], nonfinite=c['nonfinite'] == 0,
                 clean=covered,
                 **setup_gates(sessions['C'], c))
    text = (f"requests/network {c['requests']:.0f} (+{c['free_trips']:.0f} free trips); setup {c['setup_s_max']:.2f} s "
            f"(limit {LIMITS['setup_s']}; {setup_text(sessions['C'][-1]['setup'])}"
            + (f"; parent {setup_text(sessions['P'][-1]['setup'])}" if sessions['P'] else '')
            + f"); peak footprint server + client {footprint/1e9:.2f} GB (limit 7.5), group RSS {rss/1e9:.2f} GB (backstop 10); "
            f"added swap {swap/1e6:.0f} MB (limit 0); handles after {n} networks {c['peak_handles']}; paged network measurements left "
            f"out of the counters: {paged or 'none'}")
    if proj:
        gates.update(residual=proj['residual_max_s'] <= LIMITS['residual_s'], wall=proj['worst_wall_s'] <= LIMITS['worst_wall_s'])
        text += (f"; vs parent {parent_name}: kernel proxy time {proj['ratio_proxy']:.4f}x (instructions without page-fault work "
                 f"{proj['ratio_instructions_fault_free']:.4f}x; raw instructions {proj['ratio_instructions']:.4f}x, cycles "
                 f"{proj['ratio_cycles']:.4f}x, seconds {proj['ratio_kernel_s']:.4f}x), requests {proj['ratio_requests']:.4f}x, "
                 f"residual instructions {proj['ratio_residual_instructions']:.4f}x, bill {proj['ratio_flops']:.5f}x; "
                 f"PROJECTED platform wall mean {proj['mean_wall_s']:.1f} s, worst network {proj['worst_wall_s']:.1f} s "
                 f"(limit {LIMITS['worst_wall_s']:.0f}; range {proj['worst_wall_lo_s']:.1f}-{proj['worst_wall_hi_s']:.1f}), "
                 f"residual (from {proj['residual_from']}) mean {proj['residual_mean_s']:.3f} / max {proj['residual_max_s']:.3f} s (limit {LIMITS['residual_s']}); a single network may take x{proj['allow_wall']:.2f} the mean wall and x{proj['allow_residual']:.2f} the mean "
                 f"residual before a limit (S35: x1.46 and x1.31; in #333099 the slow workers' networks took x1.33-1.38 and x1.22-1.34 on average, the lost ones above x1.49 and x1.33-1.43)")
    x = stress(sessions['C'])
    if proj and x:
        # A slow worker is slow in everything: its setup binds fewer keys (measured here) AND each of its requests takes longer.
        # The projection above is this session at a normal worker's speed per request; the limits are then reached at the
        # printed factors. A worker slower than both factors loses networks; one faster than both survives even with this setup.
        first = lambda arm, k: sum(s['rows'][0][k] for s in sessions[arm])/len(sessions[arm])
        text += (f"; STRESS: the candidate sessions were set up at the pace of a worker x{x:g} slower; with that setup the "
                 f"residual limit (0.4 s) is reached by a network at x{proj['allow_residual']:.2f} a normal worker's mean time per "
                 f"request and the wall limit (120 s) at x{proj['allow_wall']:.2f}; first network after the setup against the parent's "
                 f"first under a normal setup: requests {first('C', 'requests')/first('P', 'requests'):.4f}x, residual "
                 f"instructions {first('C', 'residual_instructions')/first('P', 'residual_instructions'):.4f}x, bill "
                 f"{first('C', 'flops')/first('P', 'flops'):.5f}x")
    status, line = judge('C boundary' + (f', setup of a x{x:g} slower worker' if x else ''), gates, load, text)
    result.update(gates=gates, status=status, passed=status != 'FAIL', load_before=load, line=line, stress_slowdown=x)
    write_json(out/'stage_c.json', result)
    sys.exit(0 if status != 'FAIL' else 1)


if __name__ == '__main__':
    main()
