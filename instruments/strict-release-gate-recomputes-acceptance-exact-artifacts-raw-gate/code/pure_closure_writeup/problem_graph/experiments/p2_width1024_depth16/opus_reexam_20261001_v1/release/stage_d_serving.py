"""TAG opus-release STAGE D: the 20-network single-worker serving run. One worker session exactly as a platform worker runs it:
one flopscope server, one numpy-free client, setup() once, then 20 networks back to back with no resets (the platform gives each
of its 5 workers 20 networks; handles, caches and memory accumulate over the session). The server refuses a 70,001st array
handle (the platform's own cap is 100,000), so an overflow fails the run instead of going unnoticed.

Also checks that the production path equals the native path: networks 35 and 100 are run natively (OpenBLAS, in process, fresh
process, same order) and must give byte-identical predictions and the same bill as the worker's first two networks.
If a stored bank reference exists (calibration/bank_stage_d.json, built by make_reference.py from the bank's own stage-D run),
the session is projected onto the platform against it over the networks on which neither run paged. A reference that carries
its own platform record (make_reference.py <worker> <eval json>) is the parent of that projection instead of S35.
One PASS / PASS* / FAIL line (common.judge); limits as used for S35: all 20 x 16384 outputs finite, peak handles < 70,000, server + client
peak footprint < 7.5 GB (group RSS is only the guard's backstop at 10 GB since 2026-10-02 08:50Z), zero added
swap, setup <= 4.1 s, projected residual max <= 0.4 s, projected worst-network wall <= 105 s.
--short-setup / --setup-slowdown X: the setup runs under the platform's clock (setup_clock.py), the setup gates are then those of
stage C (`binding`, `setup_ok`), and the bank reference must have been run the same way (--reference).
usage: stage_d_serving.py <candidate.py> <tag> [--ranks ...] [--save-reference] [--no-native] [--no-gate] [--from-worker DIR]
       [--short-setup | --setup-slowdown X] [--reference FILE]
       (--save-reference: this run is the bank's own; no projection, run make_reference.py on it afterwards)"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from common import LIMITS, PLATFORM_S35, R, WORKER20, judge, peak_footprint, run_dir, sha, write_json
import project
from stage_b_native40 import native
from stage_c_boundary import load_session, session, setup_gates, setup_text, stress

REFERENCE = R/'calibration'/'bank_stage_d.json'               # the bank's 20 networks, setup under this machine's clock
REFERENCE_PACED = R/'calibration'/'bank_stage_d_paced.json'   # the same with the setup under the platform's clock (--short-setup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate', type=Path)
    ap.add_argument('tag')
    ap.add_argument('--ranks', default=','.join(map(str, WORKER20)))
    ap.add_argument('--save-reference', action='store_true', help='store this run as the bank reference (run it on the bank)')
    ap.add_argument('--no-native', action='store_true')
    ap.add_argument('--no-gate', action='store_true')
    ap.add_argument('--short-setup', action='store_true', help="the setup runs under the platform's clock (setup_clock.py)")
    ap.add_argument('--setup-slowdown', type=float, help='the same for a worker this many times slower in setup (0 = no guard)')
    ap.add_argument('--reference', type=Path, help='bank reference built by make_reference.py (default: the stored one whose '
                    'setup ran under the same clock)')
    ap.add_argument('--from-worker', type=Path, help='assess an existing worker session directory instead of running one')
    a = ap.parse_args()
    candidate = a.candidate.resolve()
    out = run_dir(a.tag)
    load = os.getloadavg()[0]
    ranks = [int(x) for x in a.ranks.split(',')]
    if a.from_worker:
        s = load_session(a.from_worker)
        assert json.loads((a.from_worker/'manifest.json').read_text())['sha256'] == sha(candidate), 'worker ran another source'
    else:
        slowdown = a.setup_slowdown if a.setup_slowdown is not None else 1.0 if a.short_setup else None
        s = session(out/'worker', candidate, a.ranks, 900 + 600*len(ranks), not a.no_gate, ['--handle-cap', str(LIMITS['handles'])]
                    + ([] if slowdown is None else ['--setup-slowdown', str(slowdown)]))
    c = project.summarize([s])
    g = s['guard']
    gates = dict(finite=c['nonfinite'] == 0 and len(s['rows']) == len(ranks), handles=c['peak_handles'] < LIMITS['handles'],
                 memory=peak_footprint([s]) < LIMITS['footprint_bytes'], swap=g['max_swap_growth_bytes'] <= LIMITS['swap_bytes'],
                 **setup_gates([s], c))
    text = (f"{len(s['rows'])} networks in one worker: outputs finite {c['nonfinite'] == 0}; peak handles {c['peak_handles']} "
            f"(limit < {LIMITS['handles']}); peak footprint server + client {peak_footprint([s])/1e9:.2f} GB (limit < 7.5), group RSS "
            f"{g['max_rss_bytes']/1e9:.2f} GB (backstop 10); added swap {g['max_swap_growth_bytes']/1e6:.0f} MB (limit 0); setup "
            f"{c['setup_s_max']:.2f} s (limit {LIMITS['setup_s']}; {setup_text(s['setup'])}); requests/network {c['requests']:.0f} "
            f"(+{c['free_trips']:.0f} free trips); tree caches at the end {s['rows'][-1]['index_cache_entries']}")
    parity = None
    if not a.no_native and ranks[:2] == [35, 100]:
        rows, where, _ = native(candidate, [35, 100], not a.no_gate)
        parity = []
        for row, mine in zip(rows, s['rows'][:2]):
            rpc = np.frombuffer((Path(s['dir'])/f"pred_{row['rank']}.bin").read_bytes(), dtype=np.float32).reshape(16, 1024)
            same = rpc.astype(np.float64).tobytes() == np.load(where/f"pred_{row['rank']}.npy").tobytes()
            parity.append(dict(rank=row['rank'], bytes_identical=same, native_flops=row['flops'], worker_flops=mine['flops']))
        gates['native_parity'] = all(x['bytes_identical'] and x['native_flops'] == x['worker_flops'] for x in parity)
        bills = ', '.join(f"{100*(x['worker_flops']/x['native_flops'] - 1):+.3f}%" for x in parity)
        text += (f"; native parity on 35,100: bytes identical {all(x['bytes_identical'] for x in parity)}, worker bill against "
                 f"native {bills} (a setup that did not bind every key bills the rest inside predictions)")
    proj = None
    paged = c['paged_ranks']
    if paged:
        text += f"; server paged on networks {paged} (their counters read high; left out of the projection)"
    if a.save_reference:
        text += '; reference: run make_reference.py on this worker directory'
    elif (a.reference or (REFERENCE if s['setup'].get('slowdown') is None else REFERENCE_PACED)).exists():
        ref = json.loads((a.reference or (REFERENCE if s['setup'].get('slowdown') is None else REFERENCE_PACED)).read_text())
        if stress([s]) is None:
            assert ref['setup'].get('slowdown') == s['setup'].get('slowdown'), 'bank reference ran its setup under another clock'
        ref_rows = {r['rank']: r for r in ref['rows'] if project.is_clean(r)}
        common = [r['rank'] for r in s['rows'] if project.is_clean(r) and r['rank'] in ref_rows]
        gates['clean'] = len(common) >= len(ranks)//2
        if common:
            keep = lambda r: r['rank'] in common
            bank = ref.get('platform')     # the reference file's own platform telemetry; a reference without one is S35's run
            proj = project.project(project.summarize([s], keep), project.summarize([dict(rows=ref['rows'], setup=ref['setup'])], keep),
                                   bank or PLATFORM_S35)
            gates.update(residual=proj['residual_max_s'] <= LIMITS['residual_s'], wall=proj['worst_wall_s'] <= LIMITS['worst_wall_s'])
            text += (f"; vs {'#' + str(bank['sid']) if bank else 'bank'} over the {len(common)} networks clean in both runs: kernel proxy time {proj['ratio_proxy']:.4f}x, "
                     f"requests {proj['ratio_requests']:.4f}x, residual instructions {proj['ratio_residual_instructions']:.4f}x, "
                     f"bill {proj['ratio_flops']:.5f}x; PROJECTED platform wall mean {proj['mean_wall_s']:.1f} s, worst network "
                     f"{proj['worst_wall_s']:.1f} s (limit {LIMITS['worst_wall_s']:.0f}; range {proj['worst_wall_lo_s']:.1f}-"
                     f"{proj['worst_wall_hi_s']:.1f}), residual (from {proj['residual_from']}) mean {proj['residual_mean_s']:.3f} / max "
                     f"{proj['residual_max_s']:.3f} s (limit {LIMITS['residual_s']}); a single network may take x{proj['allow_wall']:.2f} the mean wall and x{proj['allow_residual']:.2f} the mean "
                 f"residual before a limit (S35: x1.46 and x1.31; in #333099 the slow workers' networks took x1.33-1.38 and x1.22-1.34 on average, the lost ones above x1.49 and x1.33-1.43)")
    status, line = judge('D serving', gates, load, text)
    result = dict(stage='D', candidate=str(candidate), candidate_sha256=sha(candidate), ranks=ranks, summary=c, projection=proj,
                  max_footprint_bytes=peak_footprint([s]),
                  native_parity=parity, gates=gates, status=status, passed=status != 'FAIL', load_before=load, guard=g,
                  server=s['server'], limits=LIMITS, run=s['dir'], line=line,
                  per_network=[{k: r[k] for k in ('rank', 'flops', 'wall_s', 'kernel_s', 'kernel_instructions', 'requests',
                                                  'free_trips', 'residual_s', 'residual_instructions', 'peak_handles',
                                                  'index_cache_entries')} for r in s['rows']])
    write_json(out/'stage_d.json', result)
    sys.exit(0 if status != 'FAIL' else 1)


if __name__ == '__main__':
    main()
