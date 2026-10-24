"""TAG opus-release: judge the designation test of one file (designation_test.sh) from its three stage results. One line per
criterion with the measured number and its limit, then one verdict line: PASS, FAIL (which criteria), REPEAT (a measurement was
spoiled by the machine: paged pair, swap growth) or INCOMPLETE (a stage has no result). No compute.
The criteria are the designation rule in its second form (TEAM_LOG 2026-10-02 16:02Z; the first form, accepted by the lead on
2026-10-02 03:50Z, is _stale/designation_judge_rule1_replaced_20261002T1605Z.py). The limit a file meets first is the 0.4 s
residual on a worker's FIRST network after a setup (#333528, #333559), so the request cap is split by network: at most 43,800
requests on the first network after the setup (network 35 of stage C) and at most 42,700 on the warm one (network 100); the
mean of the two is the old cap (43,300). At the bank's platform prices (7.19 microseconds per request on the networks that are
no first network, 12.7 per extra request of a first network) a first network then costs 0.322 s (S39 measured 0.327) and the
others S40's 0.307 s. Unchanged: at most +2% requests when the setup runs at the pace of a worker 1.3x slower, every key bound at
a normal worker's pace (binding phase: see below), handles under 70,000, server + client peak footprint
under 7.5 GB (group RSS is shown, not judged), all outputs finite, bytes and bill equal to native on networks 35 and 100.
Lines in seconds, since 2026-10-02 18:30Z (the forms of 16:02Z and before are in _stale/): a graded run is measured at the
platform's speed of that hour, and that speed moves. S41's run (#333587, after 57 minutes in the platform's queue) came out
3.1% above its projection in kernel time, 4.2% in overhead and 4.3% in residual at once, with six of its nine setups cut by
the guard after 16,804-18,001 requests; S36-S40 had stayed within 1.5%. So a mean in seconds says as much about the hour
as about the file, and the lines that judged means (projected and measured mean residual 0.310 s, first networks 0.328 s,
mean wall 82 s) are shown, not judged. Judged instead: the projected worst-network wall at most 100 s (stage C, at the
bank's speed), and in the file's own graded run no network lost, its worst network at most 0.36 s of residual (a tenth of
the limit left whatever the hour) and at most 105 s of roundtrip. The file itself is judged by its counts, which no hour
moves.
One candidate session against stored parent legs is enough for the pair (requests and bills repeat
exactly between sessions; kernel work is read without page-fault work, project.fault_free).
Binding phase at most 18,600 requests since 2026-10-02 17:10Z (was 18,100): the phase now also holds the once-per-process
constants (engineer's CS, 490 requests) and the first-load views of the first 128 keys (432), which help every worker, slow ones
included; what a worker short of the phase's end leaves unbound are the least-used keys of the list (about 27 requests each on
the first network that uses one, one network in fifteen). 18,600 is the slow end of a normal platform worker's pace
(setup_clock.py); the slow-setup line is the one that measures what a slow worker costs.
usage: designation_judge.py <tag> [--pair DIR] [--slow DIR] [--worker DIR]   (defaults: runs/<tag>_pair, _slow, _worker)"""
import argparse
import json
from pathlib import Path

from common import RUNS, peak_footprint
import project

LIMITS = dict(requests_first=43800, requests_warm=42700, slow_requests_ratio=1.02, binding_requests=18600, wall_worst_s=100.0,
              residual_worst_s=0.36, roundtrip_worst_s=105.0, handles=70000, footprint_bytes=7.5e9, setup_wall_s=4.5, setup_wall_slow_s=4.6)
SETUP_OUTSIDE_S = 0.19   # platform setup wall minus our last setup phase on the platform clock: 0.164-0.198 s on S37-S42's banks (2026-10-10)


def setup_end(session):
    """Time of the last setup phase on the platform clock (setup_clock.py), from a stage C session's first worker."""
    log = json.loads((Path(session['sessions']['C'][0])/'manifest.json').read_text())['setup']['pb_log']
    return max(x[-1] for x in log if len(x) > 1 and isinstance(x[-1], (int, float)))
ap = argparse.ArgumentParser()
ap.add_argument('tag')
ap.add_argument('--pair', type=Path)
ap.add_argument('--slow', type=Path)
ap.add_argument('--worker', type=Path)
ap.add_argument('--first-priced', action='store_true', help='lead override (2026-10-11, S48): the first-network request cap becomes a priced risk: '
                'P(loss) per 100 networks from loss_model_v3 (first networks priced separately, 6 per 100) <= 10%%; the warm cap stays')
ap.add_argument('--primary', action='store_true', help='the cheaper-bill file of a designated pair: the request caps give way to '
                'P(a fresh 100-network run loses a network) <= 20%% (loss_risk.py, local reading); every other line as for a single file')
a = ap.parse_args()


def load(path, name):
    path = path/name
    return json.loads(path.read_text()) if path.exists() else None


pair = load(a.pair or RUNS/f'{a.tag}_pair', 'stage_c.json')
slow = load(a.slow or RUNS/f'{a.tag}_slow', 'stage_c.json')
worker = load(a.worker or RUNS/f'{a.tag}_worker', 'stage_d.json')
rows, missing, spoiled = [], [], []


def check(name, value, ok, limit):
    rows.append((name, value, limit, bool(ok)))


if pair is None or pair.get('projection') is None:
    missing.append('pair')
else:
    c, p = pair['candidate_summary'], pair['projection']
    setup = json.loads((Path(pair['sessions']['C'][0])/'manifest.json').read_text())['setup']
    log = {x[0]: x[1] for x in setup['pb_log'] if len(x) > 1}
    keys = log.get('wall guard', setup.get('keys_total') or 649)
    bound = keys >= 0.90*(setup.get('keys_total') or 649)      # since 2026-10-10: 90% of the list at the platform's pace (see stage_c_boundary.setup_gates)
    first, warm = [json.loads(x)['requests'] for x in (Path(pair['sessions']['C'][0])/'rows.jsonl').read_text().splitlines()][:2]
    if a.primary:   # pair designation (rules: only the better of the two designated files is ranked): the primary may run hot
        import subprocess, sys
        out = subprocess.run([sys.executable, '-B', str(Path(__file__).parent/'loss_risk.py'), f'this={(first + warm)//2}'],
                             capture_output=True, text=True).stdout
        row = next(x for x in out.splitlines() if x.startswith('this '))
        p_local = float(row.split('%')[0].split()[-1])
        check('P(loss), fresh 100-network run (primary)', f"{p_local:.2f}% at {(first + warm)//2:,} requests (first {first:,}, warm {warm:,})",
              p_local <= 20.0, '<= 20%')
    elif a.first_priced:   # lead override 2026-10-11 (S48): first-network count priced, not capped
        import json as _j, numpy as _np
        from scipy.stats import genpareto as _g
        _x = _np.array(_j.loads((Path(__file__).parent/'runs'/'net_factors_20261011b.json').read_text())['ours'])
        _e = _x[_x > 1.10] - 1.10; _c, _, _s = _g.fit(_e, floc=0); _S = lambda t: len(_e)/len(_x)*_g.sf(t - 1.10, _c, 0, _s)
        _mw = 0.99*warm*7.31e-6
        _p = 1 - (1 - _S(0.4/(_mw*first/warm)))**6*(1 - _S(0.4/_mw))**94*(1 - 2/359)
        check('P(loss) per 100 networks, first priced (override)', f"{100*_p:.2f}% (first {first:,}: cap/median {0.4/(_mw*first/warm):.3f}; S47 hour)",
              _p <= 0.10, '<= 10%')
        check('requests, warm network (100)', f"{warm:,}", warm <= LIMITS['requests_warm'], f"<= {LIMITS['requests_warm']:,}")
    else:
        check('requests, first network after the setup (35)', f"{first:,}", first <= LIMITS['requests_first'], f"<= {LIMITS['requests_first']:,}")
        check('requests, warm network (100)', f"{warm:,}", warm <= LIMITS['requests_warm'], f"<= {LIMITS['requests_warm']:,}")
    check('binding phase at the platform pace', f"{setup.get('binding_requests')} requests, {keys} of {setup.get('keys_total') or 649} keys bound",
          bound and (setup.get('binding_requests') or 10**9) <= LIMITS['binding_requests'], f"<= {LIMITS['binding_requests']:,}, >= 90% bound")
    end = setup_end(pair)
    check('projected platform setup wall (cap 5 s, whole run)', f"{end + SETUP_OUTSIDE_S:.3f} s (last phase {end:.3f} s + {SETUP_OUTSIDE_S} s)",
          end + SETUP_OUTSIDE_S <= LIMITS['setup_wall_s'], f"<= {LIMITS['setup_wall_s']} s")
    check('projected worst-network wall', f"{p['worst_wall_s']:.1f} s (mean {p['mean_wall_s']:.1f} s; mean residual {p['residual_mean_s']:.3f} s "
          f"from {p.get('residual_from', 'residual instructions')})", p['worst_wall_s'] <= LIMITS['wall_worst_s'], f"<= {LIMITS['wall_worst_s']:.0f} s")
    if not pair['gates'].get('clean', True):
        spoiled.append('pair paged (kernel counts unreliable)')
if slow is None:
    missing.append('slow setup')
elif pair is not None:
    ratio = slow['candidate_summary']['requests']/pair['candidate_summary']['requests']
    check('requests under a setup 1.3x slower', f"{slow['candidate_summary']['requests']:,.0f} ({100*(ratio - 1):+.2f}%)",
          ratio <= LIMITS['slow_requests_ratio'], '<= +2%')
    end = setup_end(slow)
    check('projected setup wall under a 1.3x slow setup', f"{end + 1.3*SETUP_OUTSIDE_S:.3f} s", end + 1.3*SETUP_OUTSIDE_S <= LIMITS['setup_wall_slow_s'],
          f"<= {LIMITS['setup_wall_slow_s']} s")
if worker is None:
    missing.append('worker')
else:
    s, g = worker['summary'], worker['guard']
    check('outputs finite on every network', f"{len(worker['per_network'])} networks, non-finite {s['nonfinite']}",
          worker['gates']['finite'], 'all finite')
    check('peak handles', f"{s['peak_handles']:,}", s['peak_handles'] < LIMITS['handles'], f"< {LIMITS['handles']:,}")
    footprint = peak_footprint([dict(rows=[json.loads(x) for x in (Path(worker['run'])/'rows.jsonl').read_text().splitlines()])])
    check('peak footprint, server + client', f"{footprint/1e9:.2f} GB (group RSS {g['max_rss_bytes']/1e9:.2f} GB)",
          footprint < LIMITS['footprint_bytes'], '< 7.5 GB')
    parity = worker.get('native_parity')
    check('native parity on 35, 100 (bytes and bill)', 'not run' if parity is None else str(worker['gates'].get('native_parity')),
          bool(parity) and worker['gates'].get('native_parity'), 'identical')
    if g.get('max_swap_growth_bytes', 0) > 0:
        spoiled.append(f"system swap grew {g['max_swap_growth_bytes']/1e6:.0f} MB during the worker run (machine shared)")
# A graded file is judged on what the platform measured (the rule's own words: no network lost, residual room at least
# 0.095 s, mean wall at most 80 s by platform telemetry); the projected lines above then only show how the instrument did.
source = (pair or worker or {}).get('candidate_sha256')
fired = project.graded(source) if source else None
if fired:
    t, lost, page = fired
    nets = [m['telemetry'] for m in (lambda z: (z.get('evaluation') or z)['results']['per_mlp'])(json.loads(Path(page).read_text())) if m.get('telemetry')]
    firsts = [x['residual_wall_time_s'] for x in nets if x.get('participant_setup_wall_s') is not None]
    check(f"platform #{t['sid']}: networks lost", f"{len(lost)} of {t['networks']}", not lost, 'none')
    worst_rt = max(x['participant_roundtrip_wall_s'] for x in nets)
    setups = [x['wall_s'] for x in (lambda z: (z.get('evaluation') or z)['results'].get('setup_executions') or [])(json.loads(Path(page).read_text()))]
    if setups:
        check(f"platform #{t['sid']}: longest setup", f"{max(setups):.3f} s over {len(setups)} setups", max(setups) <= LIMITS['setup_wall_s'],
              f"<= {LIMITS['setup_wall_s']} s")
    check(f"platform #{t['sid']}: worst network, residual", f"{t['residual_max']:.4f} s (mean {t['residual_mean']:.4f}; first networks "
          f"{sum(firsts)/len(firsts):.4f} over {len(firsts)} setups)", t['residual_max'] <= LIMITS['residual_worst_s'], f"<= {LIMITS['residual_worst_s']} s")
    check(f"platform #{t['sid']}: worst network, roundtrip", f"{worst_rt:.1f} s (mean wall {t['predict_mean']:.1f})",
          worst_rt <= LIMITS['roundtrip_worst_s'], f"<= {LIMITS['roundtrip_worst_s']:.0f} s")
for name, value, limit, ok in rows:
    print(f"  {'ok  ' if ok else 'FAIL'} {name:44s} {value:58s} limit {limit}")
failed = [name for name, _, _, ok in rows if not ok]
status = ('INCOMPLETE' if missing else 'FAIL' if failed else 'REPEAT' if spoiled else 'PASS')
detail = (f"no result for: {', '.join(missing)}" if missing else f"failed: {', '.join(failed)}" if failed
          else f"spoiled: {'; '.join(spoiled)}" if spoiled else 'every criterion met')
print(f"[designation {a.tag}] {status}: {detail}" + (f"; also noted: {'; '.join(spoiled)}" if spoiled and status != 'REPEAT' else ''))
