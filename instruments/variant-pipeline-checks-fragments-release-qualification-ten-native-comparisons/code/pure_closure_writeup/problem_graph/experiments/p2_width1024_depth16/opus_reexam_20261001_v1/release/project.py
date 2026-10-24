"""TAG opus-release: turn boundary sessions (stage C/D rows) into a projection of the platform's per-network telemetry.

The platform splits a prediction's wall into kernel (server numpy time) + overhead (flopscope machinery + wire) + residual (the
participant's Python). A candidate is projected by scaling each part of the parent's platform record (common.PLATFORM: the bank
in force, S38 #333481 since 2026-10-02; S35 #333111 before: kernel 73.41 s, overhead 8.75 s, residual 0.304 s mean / 0.339 s
max, wall 82.47 s mean / 94.93 s max) with a candidate/parent ratio measured locally on the same networks:
  kernel   : ratio of PROXY kernel time. Proxy time of an arm = sum over operation types of (instructions of that type) x
             (the best cycles-per-instruction seen for that type in either arm); the best-case rate removes the contention that
             inflates the cycles of the memory-bound adds by up to 40% in single sessions.
             PAGE-FAULT WORK IS TAKEN OUT of the instruction counts (fault_free, since 2026-10-02 09:20 UTC). When the machine
             is short of free memory it takes freed pages back, and every large array the server then allocates is paged in
             again: 0.01-0.4 million minor faults per network in a quiet window, 1-2.4 million when the box crawls, at about
             14,200 kernel instructions each. That, not CPU load, is why raw counts of ONE file read 2-10% apart between
             windows (S35, network 100: 349.8 G to 381.1 G) and why a projection needed both arms in one window (S37 was
             projected 3.7 s low from legs of two windows). Fit over 42 stored session-networks, one intercept per file and
             network: 8,900 instructions per fault in the memory-bound class, 5,300 in the CPU-bound class; what is left
             repeats to 0.27% rms (worst 0.8%) against 2.2% rms raw. With it the three fired builds project the same kernel
             time whichever window their S35 legs come from (S36 +1.4..+2.1%, S37 -0.6..+0.1%, S38 -1.9..-1.1% over four leg
             sets; raw: +0.1..+9.4%, -5.1..+3.8%, -9.2..+4.3%). Stored legs are therefore usable at any load.
             Still NOT proof against real paging: a network during which the server paged (system/user CPU above CLEAN)
             decompresses pages at a far higher price per fault; such networks are dropped from both arms (is_clean) and the
             stage asks for a rerun if too few remain.
             It assumes both arms run an operation type at the same speed per instruction (true within one kernel family; not
             when view layouts change: then judge on paired seconds in a quiet window as well).
             Memory-bound work (elementwise arithmetic, copies, reductions over large arrays: MEMORY below) and CPU-bound work
             (matmul, scatter/gather, view requests, the rest) are scaled separately, because the platform is slower on the
             former: W_MEM is the share of the bank's PLATFORM kernel time that is memory-bound. Locally it is 0.58; a
             two-class fit over four banks (calibrate.py, RESULT.md) put it at 0.80; the three builds fired since imply,
             with the page-fault work taken out and whichever S35 legs are used, 0.72-0.75 (S36), 0.62-0.66 (S37) and
             0.56-0.60 (S38), each +-0.1 because the platform repeats a run's kernel time only to 2% (S36's run was about
             1.5% fast in everything: 6.93 microseconds of residual per request against 7.03-7.07). Central 0.65 since
             2026-10-02 08:40 UTC (it was 0.72); kernel errors at 0.65: S36 +1.4..+2.1%, S37 -0.6..+0.1%, S38 -1.9..-1.1%.
             Another parent carries its own share in its platform record (set_bank.py; S38: 0.60).
  overhead : round trips (requests + hidden free trips).
  residual : requests (deterministic: the residual is paid per flopscope call, 7.0-7.3 microseconds per request on the platform
             for every bank), times the ratio of residual instructions PER REQUEST when both arms were measured calm. The
             client's residual instructions (boundary_client.py: the same wall - dispatch split in instruction units) repeat
             to +-1% when the machine is calm (44.3-45.2 thousand per request for S35) but read 5-12% high in sessions that
             crawl under load (47.7-50.8 thousand at load 25-30), so they are only used when every network of both arms
             spent fewer than CALM_CYCLES residual cycles per request (24-26 thousand in a quiet window, 38 thousand and more at load 10+).
  worst network = projected mean wall x 1.151 (the bank's own max/mean on the platform; the largest of five clean runs).
summarize(sessions) -> per-network means of one arm; project(candidate, parent) -> projection dict."""
import json
from pathlib import Path

from common import O, PLATFORM, RUNS, sha

W_MEM = (0.55, 0.65, 0.75)      # low, central, high share of memory-bound work in S35's platform kernel time
GHZ = 3.64e9                    # this machine's clock when quiet (cycles / kernel second of the bank's quietest session)
CLEAN = 0.30                    # server system/user CPU of a network above which it paged (clean networks: 0.20-0.26)
CALM_CYCLES = 30e3              # client residual cycles per request below which its residual instructions can be trusted
# (50e3 until 2026-10-02 17:12Z. Quiet-window sessions of S37-S40 spend 24.0-26.4 thousand residual cycles per request and
# 44.1-44.8 thousand residual instructions per request, the same for every file. Sessions at load 10-30 spend 38-72
# thousand cycles and read 0.4-3.5% MORE instructions per request (the fired S40 file itself: +3.5% in runs/s40_slow1124),
# which the old threshold let through as calm: against stored quiet parent legs the residual then read 1-3% high.)
FAULT_INSTR = dict(memory=8.9e3, cpu=5.3e3)   # kernel instructions per minor page fault of the server, by operation class
MATMUL = ('matmul', 'dot', 'einsum', 'tensordot', 'inner', 'vdot')
MEMORY = frozenset(('add', 'subtract', 'multiply', 'copyto', 'true_divide', 'divide', 'outer', 'astype', 'concatenate', 'copy',
                    'negative', 'maximum', 'minimum', 'sqrt', 'square', 'absolute', 'where', 'stack', 'sum', 'mean', 'clip'))


def is_clean(row):
    return row['server']['system_s'] <= CLEAN*row['server']['user_s']


def fault_free(row):
    """{operation: kernel instructions} of one network with the page-fault work taken out: each class's instructions less
    FAULT_INSTR x the server's minor faults during that network, spread over the class's operations in proportion."""
    faults, ops, out = row.get('server_minor_faults', 0), row['kernel_ops'], {}
    for cls, memory in (('memory', True), ('cpu', False)):
        mine = {k: v[3] for k, v in ops.items() if (k in MEMORY) == memory}
        total = sum(mine.values())
        scale = max(1 - FAULT_INSTR[cls]*faults/total, 0.5) if total else 1.0
        out.update({k: v*scale for k, v in mine.items()})
    return out


def summarize(sessions, keep=None):
    """Per-network means of one arm over the rows of its sessions (optionally only the rows keep(row) admits). Rows are first
    averaged per network and then over networks, so an arm that lost one network's measurement in one session (paged, dropped)
    is still compared on the same mix of networks as the other arm."""
    rows = [r for s in sessions for r in s['rows'] if keep is None or keep(r)]
    n = len(rows)
    by_rank = {}
    for r in rows:
        by_rank.setdefault(r['rank'], []).append(r)
    mean = lambda f: sum(sum(f(r) for r in group)/len(group) for group in by_rank.values())/len(by_rank)
    names = sorted({k for r in rows for k in r['kernel_ops']})
    free = {id(r): fault_free(r) for r in rows}
    op_instr = {k: mean(lambda r: free[id(r)].get(k, 0.0)) for k in names}       # page-fault work taken out
    op_cycles = {k: mean(lambda r: r['kernel_ops'].get(k, [0, 0, 0, 0])[2]) for k in names}
    op_rate = {k: min(r['kernel_ops'][k][2]/r['kernel_ops'][k][3] for r in rows
                      if k in r['kernel_ops'] and r['kernel_ops'][k][3] > 0) for k in names}
    proxy = lambda keep: sum(op_instr[k]*op_rate[k] for k in names if keep(k))/GHZ
    return dict(
        networks=n, sessions=len(sessions), ranks=[r['rank'] for r in rows], flops=mean(lambda r: r['flops']),
        paged_ranks=[r['rank'] for r in rows if not is_clean(r)],
        kernel_s=mean(lambda r: r['kernel_s']), kernel_cycles=mean(lambda r: r['kernel_cycles']),
        kernel_instructions=mean(lambda r: r['kernel_instructions']),
        kernel_instructions_fault_free=mean(lambda r: sum(free[id(r)].values())),
        minor_faults=mean(lambda r: r.get('server_minor_faults', 0)),
        op_instr=op_instr, op_cycles=op_cycles, op_rate=op_rate,
        proxy_s=proxy(lambda k: True), proxy_matmul_s=proxy(lambda k: k in MATMUL), proxy_memory_s=proxy(lambda k: k in MEMORY),
        proxy_other_s=proxy(lambda k: k not in MATMUL and k not in MEMORY),
        requests=mean(lambda r: r['requests']), free_trips=mean(lambda r: r['free_trips']),
        round_trips=mean(lambda r: r['requests'] + r['free_trips']),
        overhead_s=mean(lambda r: r['overhead_s']), residual_s=mean(lambda r: r['residual_s']),
        residual_instructions=mean(lambda r: r['residual_instructions']), residual_cycles=mean(lambda r: r['residual_cycles']),
        calm=max(r['residual_cycles']/r['requests'] for r in rows) < CALM_CYCLES,
        wall_s=mean(lambda r: r['wall_s']), wall_s_max=max(r['wall_s'] for r in rows),
        server_cycles=mean(lambda r: r['server']['cycles']),
        system_share=sum(r['server']['system_s'] for r in rows)/sum(r['server']['user_s'] for r in rows),
        kernel_ghz=sum(r['kernel_cycles'] for r in rows)/sum(r['kernel_s'] for r in rows)/1e9,
        setup_s=sum(s['setup']['setup_s'] for s in sessions)/len(sessions), setup_s_max=max(s['setup']['setup_s'] for s in sessions),
        setup_requests=max(s['setup']['requests'] for s in sessions), setup_log=sessions[-1]['setup']['pb_log'],
        peak_handles=max(r['peak_handles'] for r in rows), footprint_max=max(r['server_footprint_max'] for r in rows),
        nonfinite=sum(r['nonfinite'] for r in rows))


def platform_record(path, w_mem=None):
    """The platform's telemetry of a graded submission (its stored evaluation JSON: evaluation.results.per_mlp[].telemetry)
    as the constants project() scales, in the shape of common.PLATFORM: means over the networks, worst wall and residual.
    w_mem: this file's own (low, central, high) share of memory-bound work in its platform kernel time, when known."""
    import statistics
    record = json.load(open(path))
    results = record['evaluation']['results'] if 'evaluation' in record else record['results']
    tel = [m['telemetry'] for m in results['per_mlp'] if m.get('telemetry')]
    col = lambda k: [t[k] for t in tel if t.get(k) is not None]
    mean = lambda k: statistics.mean(col(k))
    return dict(sid=int(record.get('submission_id') or record['evaluation']['submission_id']), networks=len(tel),
                predict_mean=mean('participant_predict_wall_s'), predict_max=max(col('participant_predict_wall_s')),
                kernel_mean=mean('flopscope_server_kernel_wall_s'), overhead_mean=mean('flopscope_server_overhead_wall_s'),
                residual_mean=mean('residual_wall_time_s'), residual_max=max(col('residual_wall_time_s')),
                flops_mean=mean('flops_used'), **({'w_mem': list(w_mem)} if w_mem else {}))


def graded(digest):
    """(platform record, lost networks, evaluation path) of the file if the lead has fired it and the grading is complete:
    a stage-A archive of this source -> the fire receipt with that archive's sha256 (O/fire/*/receipt.json) -> the stored
    evaluation of that submission. None when it was not fired from one of this lane's archives or is not graded yet."""
    for a in RUNS.glob('a_*/stage_a.json'):
        d = json.loads(a.read_text())
        archive = Path(d.get('archive') or '')
        if d.get('candidate_sha256') != digest or not archive.is_file():
            continue
        packed = sha(archive)
        for receipt in (O/'fire').glob('*/receipt.json'):
            r = json.loads(receipt.read_text())
            ev = receipt.parent/f"eval_{r.get('submission_id')}.json"
            if r.get('sha256') == packed and ev.exists():
                e = json.loads(ev.read_text())['evaluation']
                if e.get('state') == 'completed':
                    lost = [m['mlp_index'] for m in e['results']['per_mlp'] if m.get('error_code') not in (None, 'SUCCESS')]
                    return platform_record(ev), lost, ev
    return None


def proxy(arm, rate, keep):
    """Proxy kernel seconds of an arm under shared best-case rates."""
    return sum(v*rate[k] for k, v in arm['op_instr'].items() if keep(k))/GHZ


def project(c, p, platform=PLATFORM):
    rate = {k: min(x for x in (c['op_rate'].get(k), p['op_rate'].get(k)) if x is not None)
            for k in set(c['op_rate']) | set(p['op_rate'])}
    is_mem = lambda k: k in MEMORY
    r_mem = proxy(c, rate, is_mem)/proxy(p, rate, is_mem)
    r_cpu = proxy(c, rate, lambda k: not is_mem(k))/proxy(p, rate, lambda k: not is_mem(k))
    w_mem = tuple(platform.get('w_mem') or W_MEM)     # W_MEM is S35's share; another parent carries its own (set_bank.py)
    kernel = [platform['kernel_mean']*(w*r_mem + (1 - w)*r_cpu) for w in w_mem]
    r_rt = c['round_trips']/p['round_trips']
    overhead = platform['overhead_mean']*r_rt
    r_instr = c['residual_instructions']/p['residual_instructions']
    calm = bool(c.get('calm') and p.get('calm'))
    r_res = r_instr if calm else c['requests']/p['requests']
    residual = platform['residual_mean']*r_res
    walls = [k + overhead + residual for k in kernel]
    worst = platform['predict_max']/platform['predict_mean']
    residual_max = residual*platform['residual_max']/platform['residual_mean']
    return dict(
        ratio_flops=c['flops']/p['flops'], ratio_proxy=proxy(c, rate, lambda k: True)/proxy(p, rate, lambda k: True),
        ratio_proxy_memory=r_mem, ratio_proxy_cpu=r_cpu,
        ratio_instructions=c['kernel_instructions']/p['kernel_instructions'],
        ratio_instructions_fault_free=c['kernel_instructions_fault_free']/p['kernel_instructions_fault_free'],
        ratio_cycles=c['kernel_cycles']/p['kernel_cycles'], ratio_kernel_s=c['kernel_s']/p['kernel_s'],
        ratio_round_trips=r_rt, ratio_requests=c['requests']/p['requests'], ratio_residual_instructions=r_instr,
        ratio_residual=r_res, residual_from='residual instructions' if calm else 'requests',
        ratio_residual_s=c['residual_s']/p['residual_s'], ratio_overhead_s=c['overhead_s']/p['overhead_s'],
        kernel_s=kernel[1], kernel_lo_s=min(kernel), kernel_hi_s=max(kernel), overhead_s=overhead,
        residual_mean_s=residual, residual_max_s=residual_max,
        mean_wall_s=walls[1], worst_wall_s=walls[1]*worst, worst_wall_lo_s=min(walls)*worst, worst_wall_hi_s=max(walls)*worst,
        # how much slower than the bank's platform run a worker may get before a limit is hit on its worst network (120 s wall,
        # 0.4 s residual). For scale: the slow workers of the failed #333099 ran wall x1.34 and residual x1.21-1.30.
        headroom_wall=120/(walls[1]*worst), headroom_residual=0.4/residual_max,
        # the same against the MEAN of normal workers: how many times the run's mean a single network may take before a limit.
        # S35: 1.455 (wall), 1.315 (residual). In #333099 the two slow workers' networks took 1.33-1.38 (wall) and 1.22-1.34
        # (residual) times the normal mean on average; the four lost ones were above 1.49 (wall) and at 1.33-1.43 (residual).
        allow_wall=120/walls[1], allow_residual=0.4/residual, platform_reference=platform, w_mem=w_mem)
