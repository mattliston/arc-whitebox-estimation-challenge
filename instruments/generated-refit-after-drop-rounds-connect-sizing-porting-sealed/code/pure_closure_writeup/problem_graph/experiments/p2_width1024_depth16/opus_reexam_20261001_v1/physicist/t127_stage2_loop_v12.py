#!/usr/bin/env python3
"""TAG opus-physicist T127 loop v12 (2026-10-13 10:5xZ) = v10 with the STARTED line's format fixed (v10 crashed there at S54, 10:45:46Z, before any chain:
its second format tuple repeated (n, v.name), a latent bug since v1 that no round had reached); it CONTINUES v10's state file t127_state_v10.json
(S52 skipped, S53 cancelled, S54 staged); own log / stop (t127_loop_v12.log, STOP_T127L).
T127 loop v10 (2026-10-13 00:0xZ; trainer 10-12 23:5xZ) = v8 with the WORKER chosen for the round's new per-layer families
(pe3 / pe4 need worker_generated_v14, k21k v15: v13 asserts on unknown families) and launchers matched in cert/RAD and cert/Nh; otr_pl*.py
added to the judge's pinned inventory; own state / log / stop (t127_state_v10.json, t127_loop_v10.log, STOP_T127J); replaces the waiting v8.
T127 loop v8 (2026-10-12 23:0xZ) = v6 with the baseline = the LAST FIRED POINT UNDER THE ROUND'S OWN CONFIG: the round's start-point
request a_RS<N>C_ab with its point fields (params, hu) taken from a_RS<M>C_pt, M = the round of the newest O/fire/<tag> with a receipt, TWINS INCLUDED.
Why: v6 took a_RS<M>C_pt as it was, from the newest fire without 'twin'. The fired twin s51k1twin (22:35Z) = S51 + gk, and from S53 every request carries
gk (ABSORB gk11), so v6 would have judged S53 + gk against S51 without gk, crediting gk's -0.35% to S53. When the round started from the last fired
point (S53 from S51's) the baseline is a_RS<N>C_ab itself and the trainer's stage-1 pool stays on. Own state / log / stop (t127_state_v8.json,
t127_loop_v8.log, STOP_T127H); replaces the waiting v6 (no round started).
T127 loop v6 (lead 16:5xZ relaxed RAD acceptance) = v4 with the BASELINE = the frozen point of the round behind the LAST FIRED safe file
(newest O/fire/<tag> without 'twin' with a receipt; S50 = #335508 today) instead of the round's start point; the trainer's stage-1 pool only when the two
baselines are the same request; own state / log / stop (t127_state_v6.json, t127_loop_v6.log, STOP_T127F).
T127 loop v4 (lead 15:2xZ disk) = v2 for S52-S63 with the chains on t112_chain_v17.sh (each lane's weight cache -> one directory per run under
physicist/sealed_cache/), its own state / log / stop (t127_state_v4.json, t127_loop_v4.log, STOP_T127D); replaced the waiting v2 before any round started.
T127 generic stage-2 loop v2 (= v1 with the chains on t112_chain_v16.sh = the release guard v2, 08:2xZ)
v1 = TAG opus-physicist T127 generic stage-2 loop v1 (lead 10-12 08:0xZ: ONE trigger for the trainer's RAD rounds S52-S63; S51 stays on t126_trigger.sh).
Rounds in order. Round N:
  1. waits until trainer/cert/RAD/S<N>/CHECK_VERDICT_v1.json or RESERVE_VERDICT_v1.json has "accepted": true (the verdict FILES, never log wording);
     skipped if its FROZEN_POINT_v1.json is null, or once a LATER round has a CHECK_VERDICT_v1.json (the round is over), or at the deadline;
  2. re-pins the judge: the worker = the highest worker_generated_v<k>.py named by the launchers the round's chain scripts use (cert/RAD/launch_gen_*.sh);
     writes physicist/t85_pinned_judge_S<N>.py = t85_pinned_judge_v14.py with that worker + sha, the round's sealed half as FINAL_SEEDS, and every
     trainer worker_generated_v*.py / otr_m_v*.py added to the source inventory;
  3. stages the trainer's check claims a_RS<N>C_ab / a_RS<N>C_pt (id / rank / truth_plan removed) with the round's FROZEN_POINT_v1.json as the receipt;
  4. waits until all 32 sealed truths are ready (12 h at most);
  5. runs 2 chains of 16 (t112_chain_v15.sh: the machine gate before every seed), then re-serves cut lanes (archived attempt + guard record) up to 3 times;
  6. posts '[physicist] <stamp> S<N> STAGE 2 TABLE: sealed <x>% pooled (se <se>), <k> of 32 lower | ...' (physicist/t112_table.py).
Progress per round in physicist/t127_state.json (a restart resumes); private log physicist/t127_loop.log. Stop: physicist/STOP_T127.
usage: t127_stage2_loop_v1.py            (the loop)
       t127_stage2_loop_v1.py --dry <N>  (judge generation + prepare for round N on the dry claims; nothing runs)"""
import calendar
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE.parent
T = O / 'trainer'
RAD = T / 'cert' / 'RAD'
PY = '/Users/matt/anaconda3/bin/python'
ROUNDS = {52: 7101128, 53: 7101320, 54: 7101512, 55: 7101896, 56: 7102280, 57: 7102664, 58: 7103048, 59: 7103432, 60: 7103816,
          61: 7104200, 62: 7104584, 63: 7104968}
TEMPLATE = HERE / 't85_pinned_judge_v14.py'
END = calendar.timegm((2026, 10, 17, 23, 59, 0, 0, 0, 0))
STOP, STATE, NOTE = HERE / 'STOP_T127L', HERE / 't127_state_v10.json', HERE / 't127_loop_v12.log'
POINT = ('params', 'hu')      # v8: the request fields a RAD frozen point owns (FROZEN_POINT_v1.json: params + head); everything else is the round's config


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def note(msg):
    with NOTE.open('a') as fh:
        fh.write('%s %s\n' % (stamp(), msg))


def post(msg):
    with (O / 'TEAM_LOG.md').open('a') as fh:
        fh.write('[physicist] %s %s\n' % (stamp(), msg))
    note('POSTED ' + msg[:200])


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state):
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)


def accepted_file(n):
    for name in ('CHECK_VERDICT_v1.json', 'RESERVE_VERDICT_v1.json'):
        f = RAD / ('S%d' % n) / name
        try:
            if f.exists() and json.loads(f.read_text()).get('accepted') is True:
                return f
        except (ValueError, OSError):
            pass
    return None


def null_round(n):
    f = RAD / ('S%d' % n) / 'FROZEN_POINT_v1.json'
    try:
        return f.exists() and json.loads(f.read_text()).get('null') is True
    except (ValueError, OSError):
        return False


def superseded(n):
    return any((RAD / ('S%d' % m) / 'CHECK_VERDICT_v1.json').exists() for m in range(n + 1, 70))


CERT = HERE / 't127_certified_workers.json'      # v10: {'versions': [...]} the trainer-certified worker versions, read at every round
NEED = {'pe3': 14, 'pe4': 14, 'k21k': 15}      # v10: the first trainer worker whose card / otr_pl knows these per-layer families (trainer 10-12 23:5xZ)


def new_families(n):
    """v10: the new per-layer families round n can carry: RULES_v1.json new_families_t, keep_v1.json arms (dpe3_10_11, dk21k_12_14, ...), check claims' pl."""
    found = set()
    d = RAD / ('S%d' % n)
    try:
        r = json.loads((d / 'RULES_v1.json').read_text()) if (d / 'RULES_v1.json').exists() else {}
        found |= {f for f in NEED if f in json.dumps(r.get('new_families_t') or {})}
        k = json.loads((d / 'keep_v1.json').read_text()) if (d / 'keep_v1.json').exists() else {}
        found |= {f for f in NEED for a in (k.get('keep') or []) if str(a).startswith('d' + f + '_') or str(a) == f}
    except (ValueError, OSError):
        pass
    for c in (T / ('run_RAD_S%d_gen' % n) / 'claimed').glob('a_RS%dC_*_r*.w*.json' % n):
        try:
            found |= {f for f in NEED if f in (json.loads(c.read_text()).get('pl') or {})}
        except (ValueError, OSError):
            pass
    return found


def worker_for(n):
    """v10: the highest worker named by ANY launcher the round's scripts use (launch_gen_* in cert/RAD, launch_worker_* in cert/Nh), raised to the
    first worker that knows the round's new families; an absent required worker raises (the round then posts NOT STARTED)."""
    text = ' '.join(p.read_text(errors='ignore') for p in (RAD / ('S%d' % n)).glob('*.sh'))
    versions = set()
    todo, seen = set(re.findall(r'launch_(?:gen|worker)_v\d+\w*\.sh', text)), set()
    while todo:                 # follow launcher -> launcher indirections (cert/RAD/launch_gen_* -> cert/Nh/launch_worker_* -> worker_generated_v*.py)
        launcher = todo.pop(); seen.add(launcher)
        for f in (RAD / launcher, RAD.parent / 'Nh' / launcher):
            if f.exists():
                body = f.read_text(errors='ignore')
                versions |= {int(v) for v in re.findall(r'worker_generated_v(\d+)\.py', body)}
                todo |= set(re.findall(r'launch_(?:gen|worker)_v\d+\w*\.sh', body)) - seen
    if not versions:
        versions = {int(p.stem.split('_v')[1]) for p in T.glob('worker_generated_v*.py') if p.stem.split('_v')[1].isdigit()}
    need = max([NEED[f] for f in new_families(n)] or [0])
    cert = max(json.loads(CERT.read_text())['versions'])      # v10: only workers the trainer certified (bit for bit at the new families' 0) judge
    v = max([x for x in versions if x <= cert] + [need])
    if need > cert:
        raise RuntimeError('round S%d carries %s and needs worker v%d, not yet certified (physicist/%s lists %s)' % (n, sorted(new_families(n)), need, CERT.name, cert))
    w = T / ('worker_generated_v%d.py' % v)
    if not w.exists():
        raise RuntimeError('round S%d needs worker v%d, which does not exist' % (n, v))
    return w


def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_judge(n, first, worker, tag=''):
    s = TEMPLATE.read_text()

    def sub(pattern, repl):
        nonlocal s
        s2, k = re.subn(pattern, repl, s, flags=re.M)
        assert k == 1, (pattern, k)
        s = s2
    sub(r"^FINAL_SEEDS = list\(range\(\d+, \d+\)\).*$", "FINAL_SEEDS = list(range(%d, %d))   # T127: S%d's sealed half" % (first, first + 32, n))
    sub(r"^WORKER = TRAINER / 'worker_generated_v\d+\.py'.*\nWORKER_SHA = '[0-9a-f]{64}'$",
        "WORKER = TRAINER / '%s'   # T127: the worker of S%d's launchers\nWORKER_SHA = '%s'" % (worker.name, n, sha256(worker)))
    sub(r"^    files\.update\(\(O / 'engineer/a33_starsum'\)\.glob\('\*\.py'\)\)$",
        "    files.update((O / 'engineer/a33_starsum').glob('*.py'))\n    files.update(p.resolve() for p in TRAINER.glob('worker_generated_v*.py'))   # T127\n"
        "    files.update(p.resolve() for p in TRAINER.glob('otr_m_v*.py'))\n    files.update(p.resolve() for p in TRAINER.glob('otr_pl*.py'))   # T127 v10")
    sub(r"root_seed_assignment='[^']*',", "root_seed_assignment='S%d sealed half (legacy truths): %d-%d, the physicist\\'s stage 2',"
        % (n, first, first + 31))
    s = s.replace('"""Immutable paired judging', '"""T127 S%d (generated %s from t85_pinned_judge_v14.py: sealed %d-%d, worker %s).\nImmutable paired judging'
                  % (n, stamp(), first, first + 31, worker.name), 1)
    out = HERE / ('t85_pinned_judge_S%d%s.py' % (n, tag))
    k = 1
    while out.exists() and out.read_text() != s:
        k += 1
        out = HERE / ('t85_pinned_judge_S%d%s_v%d.py' % (n, tag, k))
    if not out.exists():
        out.write_text(s)
    return out


def last_fired():
    """(round, tag) of the newest O/fire/<tag> that has a receipt.json (v8: twins included), the round parsed from the tag ('s51k1twin_20261012' -> 51)."""
    best = None
    for d in (O / 'fire').iterdir():
        r = d / 'receipt.json'
        if not d.is_dir() or not r.exists():
            continue
        m = re.match(r's(\d+)', d.name)
        if not m:
            continue
        try:
            utc = json.loads(r.read_text()).get('utc', '')
        except ValueError:
            continue
        if best is None or utc > best[0]:
            best = (utc, int(m.group(1)), d.name)
    return (best[1], best[2]) if best else None


def baseline_for(n, ab, lf):
    """v8: (request file, note) of the baseline = the last fired point under round n's own config: a_RS<n>C_ab with its point fields (POINT) taken from
    a_RS<M>C_pt of the last fired round M; a_RS<n>C_ab itself when the point already agrees (params compared by file sha, hu by value)."""
    if not lf:
        return ab, 'the round\'s start point (a_RS%dC_ab; nothing fired)' % n
    fp = sorted((T / ('run_RAD_S%d_gen' % lf[0]) / 'claimed').glob('a_RS%dC_pt_r*.w*.json' % lf[0]))
    if not fp:
        return ab, 'the round\'s start point (FALLBACK: last fired %s has no RAD claims)' % lf[1]
    a, f = json.loads(Path(ab).read_text()), json.loads(fp[0].read_text())
    same = lambda k: a.get(k) == f.get(k) or (k == 'params' and a.get(k) and f.get(k) and sha256(a[k]) == sha256(f[k]))
    cfg = ', '.join(sorted(k for k in set(a) | set(f) if k not in ('id', 'rank', 'truth_plan') + POINT and a.get(k) != f.get(k))) or 'nothing'
    if all(same(k) for k in POINT):
        return ab, ('last fired S%d (%s) point = the round\'s start point a_RS%dC_ab (config differs from a_RS%dC_pt in: %s)'
                    % (lf[0], lf[1], n, lf[0], cfg))
    b = dict(a, **{k: f[k] for k in POINT if k in f})
    out = HERE / ('t127_S%d_lastfired_base.json' % n)
    k = 1
    while out.exists() and json.loads(out.read_text()) != b:
        k += 1
        out = HERE / ('t127_S%d_lastfired_base_v%d.json' % (n, k))
    if not out.exists():
        out.write_text(json.dumps(b, indent=1, sort_keys=True))
    return out, ('last fired S%d (%s) point (params, hu of a_RS%dC_pt) under S%d\'s config (a_RS%dC_ab; config differs from a_RS%dC_pt in: %s)'
                 % (lf[0], lf[1], lf[0], n, n, lf[0], cfg))


def stage(n, judge, ab, pt, receipt, prefix, run_name):
    copies = []
    for role, src in (('base', ab), ('cand', pt)):
        r = json.loads(Path(src).read_text())
        for key in ('id', 'rank', 'truth_plan'):
            r.pop(key, None)
        dst = HERE / ('%s_%s_request.json' % (prefix, role))
        k = 1
        while dst.exists() and json.loads(dst.read_text()) != r:
            k += 1
            dst = HERE / ('%s_%s_request_v%d.json' % (prefix, role, k))
        if not dst.exists():
            dst.write_text(json.dumps(r, indent=1, sort_keys=True))
        copies.append(dst)
    run = HERE / run_name
    k = 1
    while run.exists():
        k += 1
        run = HERE / ('%s_v%d' % (run_name, k))
    cmd = [PY, '-B', str(judge), 'prepare', '--baseline', str(copies[0]), '--candidate', str(copies[1]), '--out', str(run), '--purpose', 'judge',
           '--decision-receipt', str(receipt), '--threads', '3']
    p = subprocess.run(cmd, capture_output=True, text=True)
    note('prepare S%d rc %d: %s %s' % (n, p.returncode, p.stdout[-300:], p.stderr[-500:]))
    if p.returncode != 0:
        raise RuntimeError('prepare failed: ' + (p.stderr.strip().splitlines() or ['?'])[-1])
    return run / 'manifest.json'


def ready(judge, man, first):
    p = subprocess.run([PY, '-B', str(judge), 'ready', str(man), str(first)], capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])['ready'] is True
    except (ValueError, IndexError, KeyError):
        return False


def run_chains(judge, man, groups, names):
    logs = []
    for seeds, name in zip(groups, names):
        lg = HERE / ('t112_chain_%s.log' % name)
        k = 1
        while lg.exists():          # never reuse a chain log: a stale CHAINDONE would end the wait at once
            k += 1
            name = '%s_v%d' % (name.rsplit('_v', 1)[0] if '_v' in name else name, k)
            lg = HERE / ('t112_chain_%s.log' % name)
        subprocess.Popen(['bash', str(HERE / 't112_chain_v17.sh'), str(judge), str(man), name] + [str(s) for s in seeds],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        logs.append(lg)
        note('chain %s started on %d seeds' % (name, len(seeds)))
    while not all(lg.exists() and 'CHAINDONE' in lg.read_text() for lg in logs):
        if STOP.exists():
            note('STOP_T127 while chains run (chains keep their own STOP_T112)')
        time.sleep(60)


def incomplete(run, seeds):
    return [s for s in seeds if not (run / ('seed%d' % s) / 'COMPLETE.json').exists()]


def archive(n, run, s, k):
    lane = run / ('seed%d' % s)
    arc = lane / ('attempt%d_cut_%s' % (k, stamp().replace('-', '').replace(':', '')))
    (arc / 'results').mkdir(parents=True)
    for f in list((lane / 'results').iterdir()):
        f.rename(arc / 'results' / f.name)
    for name in ('STARTED.json', 'worker.log'):
        if (lane / name).exists():
            (lane / name).rename(arc / name)
    g = HERE / 't112_guards' / ('seed%d.json' % s)
    if g.exists():
        gd = HERE / 't112_guards' / ('S%d_attempt%d' % (n, k))
        gd.mkdir(exist_ok=True)
        g.rename(gd / g.name)


def table(n, run, pool=True):
    stage1 = T / ('run_RAD_S%d_gen' % n) / 'results'
    args = [PY, '-B', str(HERE / 't112_table.py'), str(run)]
    if pool and stage1.exists() and list(stage1.glob('a_RS%dC_ab_r*.json' % n)):
        args += [str(stage1), 'a_RS%dC_ab' % n, 'a_RS%dC_pt' % n]
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0 and len(args) > 4:
        p = subprocess.run(args[:4], capture_output=True, text=True)
    note('table S%d rc %d:\n%s%s' % (n, p.returncode, p.stdout, p.stderr[-500:]))
    tail = r'\s+n (\d+): final ([+-]?\d+\.\d+)% pooled \(SE of per-network % changes (\d+\.\d+)%.*?lower on (\d+) of (\d+); all-layer ([+-]?\d+\.\d+)%'
    m = re.search(r'stage 2 \(sealed\)' + tail, p.stdout)
    if not m:
        raise RuntimeError('table unparsed (rc %d)' % p.returncode)
    q = re.search(r'pooled stages 1 \+ 2' + tail, p.stdout)
    return m, q


def run_round(n, state):
    first = ROUNDS[n]
    seeds = list(range(first, first + 32))
    st = state.setdefault(str(n), {})
    while True:
        if STOP.exists():
            note('STOP_T127: exit')
            sys.exit(0)
        if time.time() > END:
            note('deadline: exit')
            sys.exit(0)
        v = accepted_file(n)
        if v:
            break
        if null_round(n):
            st.update(done=True, outcome='skipped: frozen point null'); save_state(state); note('S%d skipped (null)' % n); return
        if superseded(n):
            st.update(done=True, outcome='skipped: a later round has a check verdict'); save_state(state); note('S%d skipped (superseded)' % n); return
        time.sleep(60)
    note('S%d accepted by %s' % (n, v.name))
    claims = T / ('run_RAD_S%d_gen' % n) / 'claimed'
    ab = sorted(claims.glob('a_RS%dC_ab_r*.w*.json' % n))
    pt = sorted(claims.glob('a_RS%dC_pt_r*.w*.json' % n))
    receipt = RAD / ('S%d' % n) / 'FROZEN_POINT_v1.json'
    try:
        if not (ab and pt and receipt.exists()):
            raise RuntimeError('check claims a_RS%dC_ab / _pt or FROZEN_POINT_v1.json missing' % n)
        worker = worker_for(n)      # v10: inside the try, so a missing required worker posts NOT STARTED instead of ending the loop
        judge = make_judge(n, first, worker)
        base_claim, base_note = baseline_for(n, ab[0], last_fired())
        st['baseline'] = base_note
        strip = lambda f: {k: v for k, v in json.loads(Path(f).read_text()).items() if k not in ('id', 'rank', 'truth_plan')}
        st['pool_with_stage1'] = strip(base_claim) == strip(ab[0])
        man = stage(n, judge, base_claim, pt[0], receipt, 't127_S%d' % n, 't127_S%d_stage2' % n)
    except Exception as e:
        post('S%d STAGE 2 NOT STARTED: %s (verdict %s accepted); waiting for the lead.' % (n, e, v.name))
        st.update(done=True, outcome='not started: %s' % e); save_state(state); return
    run = man.parent
    st.update(judge=str(judge), worker=worker.name, run=str(run)); save_state(state)
    t0 = time.time()
    while not ready(judge, man, first):
        if STOP.exists() or time.time() > END:
            sys.exit(0)
        if time.time() - t0 > 12 * 3600:
            post('S%d STAGE 2 NOT STARTED: the 32 sealed truths %d-%d were not ready within 12 h.' % (n, first, first + 31))
            st.update(done=True, outcome='not started: truths'); save_state(state); return
        time.sleep(120)
    post('S%d STAGE 2 STARTED (verdict %s accepted = true; baseline = ' % (n, v.name) + st.get('baseline', '?') + '): judge %s (worker %s, sha-pinned) on SEALED %d-%d, base %s vs candidate %s (frozen point), 2 chains '
         'of 16 with the machine gate before each seed (physicist/t112_chain_v17.sh), run %s.' % (judge.name, worker.name, first, first + 31,   # v12: the tuple
                                                                                                 Path(base_claim).name, pt[0].name, run.relative_to(O)))   # repeated (n, v.name)
    run_chains(judge, man, [seeds[:16], seeds[16:]], ['S%da' % n, 'S%db' % n])
    for k in (1, 2, 3):
        cut = incomplete(run, seeds)
        if not cut:
            break
        note('S%d: %d lanes incomplete after pass %d: %s' % (n, len(cut), k, cut))
        for s in cut:
            archive(n, run, s, k)
        run_chains(judge, man, [cut], ['S%dr%d' % (n, k)])
    cut = incomplete(run, seeds)
    if cut:
        post('S%d STAGE 2 INCOMPLETE after 3 re-serves: %d of 32 lanes without COMPLETE.json (%s); no table.' % (n, len(cut), ' '.join(map(str, cut))))
        st.update(done=True, outcome='incomplete'); save_state(state); return
    try:
        m, q = table(n, run, pool=st.get('pool_with_stage1', True))
    except Exception as e:
        post('S%d STAGE 2 complete (32 of 32) but the table failed: %s; run %s.' % (n, e, run.relative_to(O)))
        st.update(done=True, outcome='table failed'); save_state(state); return
    extra = ' | all-layer %s%%' % m.group(6)
    if q:
        extra += "; pooled with the trainer's check %s%% (se %s, %s of %s lower)" % (q.group(2), q.group(3), q.group(4), q.group(5))
    extra += '; baseline %s' % st.get('baseline', '?')
    post('S%d STAGE 2 TABLE: sealed %s%% pooled (se %s), %s of %s lower%s; run %s.' % (n, m.group(2), m.group(3), m.group(4), m.group(5), extra,
                                                                                    run.relative_to(O)))
    st.update(done=True, outcome='table', sealed=float(m.group(2)), se=float(m.group(3)), lower=int(m.group(4))); save_state(state)


def dry(n):
    first = ROUNDS[n]
    worker = worker_for(n)
    judge = make_judge(n, first, worker, tag='_DRYTEST_unused')
    man = stage(n, judge, HERE / 't117v12_DRY_base_claim.json', HERE / 't117v12_DRY_cand_claim.json', T / 'cert/RAD/MS/FROZEN_POINT_v1.json',
                't127_S%d_DRYTEST_unused' % n, 't127_S%d_DRYTEST_unused_run' % n)
    m = json.loads(man.read_text())
    print('dry S%d: judge %s, worker %s, seeds %d-%d (%d), manifest %s' % (n, judge.name, worker.name, m['seeds'][0], m['seeds'][-1], len(m['seeds']), man))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--dry':
        dry(int(sys.argv[2]))
        sys.exit(0)
    state = load_state()
    note('loop start; state %s' % json.dumps(state))
    for n in sorted(ROUNDS):
        if state.get(str(n), {}).get('done'):
            continue
        run_round(n, state)
    note('all rounds done')
