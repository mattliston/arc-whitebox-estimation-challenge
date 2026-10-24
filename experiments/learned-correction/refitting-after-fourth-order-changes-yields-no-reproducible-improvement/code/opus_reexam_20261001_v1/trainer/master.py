"""TAG opus-trainer (2026-10-01): master of the parallel long-horizon refit. Owns the parameters (28 arrays + 12 dials of the research
parent RRPI; 23 arrays and 10 dials trainable), Adam and an exponential moving average (EMA) of the trainable parameters. Workers
(worker.py, kept alive by supervisor.py) only compute per-network gradients and forward evaluations; all traffic is files in the run dir.
One update: publish params/p<k>.pt (the parameters after k updates), queue one gradient request per network of the batch with the
loss scaled 1e8/batch, add the per-network gradients in batch order, clip the summed gradient (arrays and dials separately, max norm
--clip, as the reference trainer), Adam step, EMA step. Learning rates: full / trunk / per-dial groups as the reference, linear warm-up
over --warm updates, then constant. Training ranks: the 641 of split_train640 'train', reshuffled every epoch (seeded); batches are
cut from the concatenated epochs and a trailing partial batch is not run, so raising --epochs later only appends updates.
Every --eval-every updates (and at 0 and at the end): forward-only final-layer MSE on VAL-A (20 nets) for the raw and the EMA
parameters. Every --ckpt-every updates: ckpt/after_<k>.pt with parameters, EMA and optimizer state. Starting the same command again
resumes from the newest checkpoint; updates between that checkpoint and the crash are replayed from the gradient files already on
disk and must reproduce the parameter files on disk bit for bit. Nothing is ever deleted or overwritten.
A run can continue another run's checkpoint (--init: parameters, EMA, Adam state; warm-up and EMA schedule count on from
there) with its own training ranks (--train-file) and its own validation ranks kept out of training (--val-file); with
--eval-what ema only the EMA parameters are scored (the untrained parameters are scored once as the reference).
--extra family:lr,... adds GLOBAL trainable physics dials beyond the 12: one shift per family (otr_pl.py), applied on every
layer, 0 = today's value; it needs workers started with TRAINER_PL=1; the shifts travel in the parameter files ('pl').
--extra-init family:shift,...: start values of extra dials (shift from today's value). When any is non-zero, or with --init, the
untrained parameters with no shift are scored once at the start as the reference ('untrained'), and the start point itself is
not scored at update 0 unless --eval-what is left at its default.
--dial-lr dial:rate,...: learning rates of single dials of the 10 (in place of the reference trainer's times --dial-lr-scale).
--dial-init dial:shift,...: start values of the 10 dials as shifts from the parent's (the reference is then scored as above).
--card M: every result must come from a worker running the run M card (TRAINER_CARD=M: F4 + K4L, otr_m.py); without it, from today's.
<run>/STAR_ATTACH containing 1 (read before every update): gradient requests ask for the star basis attached in the backward pass.
usage: nohup /Users/matt/anaconda3/bin/python -B master.py <run dir> [--epochs 2 ...] >> <run dir>/master.out 2>&1 &
watch: tail -F <run dir>/progress.txt      stop: touch <run dir>/STOP (workers and master leave; resume later with the same command)"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from otr_common import (DIAL_LR, FROZEN_DIALS, FROZEN_P, SOURCE_SHA, TRUNK, enqueue, load_params, load_pl, publish, read_source,
                        result_path, run_dirs, save_params, splits, wait_results)

parser = argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch', type=int, default=12)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr-full', type=float, default=1e-4)
parser.add_argument('--lr-trunk', type=float, default=5e-5)
parser.add_argument('--dial-lr-scale', type=float, default=1.0)
parser.add_argument('--warm', type=int, default=10)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--ema', type=float, default=0.95)
parser.add_argument('--eval-every', type=int, default=18)
parser.add_argument('--ckpt-every', type=int, default=25)
parser.add_argument('--smoke', action='store_true', help='plumbing test: 3 updates, 2 validation nets')
parser.add_argument('--init', default=None, help='checkpoint of another run (ckpt/after_<k>.pt) to continue from')
parser.add_argument('--train-file', default=None, help='file with comma-separated training ranks (default: the 641)')
parser.add_argument('--val-file', default=None, help='file with comma-separated validation ranks (default: VAL-A)')
parser.add_argument('--eval-what', default=None, choices=('ema',), help='ema: score only the EMA parameters')
parser.add_argument('--init-updates', type=int, default=None, help='updates behind the --init checkpoint (default: its own count)')
parser.add_argument('--extra', default=None, help='extra GLOBAL trainable physics dials, family:lr,... (families of otr_pl.py)')
parser.add_argument('--card', default=None, choices=('M',), help='M: the workers must run the run M card (TRAINER_CARD=M)')
parser.add_argument('--extra-init', default=None, help='start shifts of extra dials, family:shift,... (default 0 = today)')
parser.add_argument('--dial-lr', default=None, help='learning rates of single dials, dial:rate,...')
parser.add_argument('--dial-init', default=None, help='start shifts of the 10 trainable dials, dial:shift,...')
args = parser.parse_args()
RUN = run_dirs(Path(args.run).resolve())
config = {key: value for key, value in vars(args).items() if key not in ('run', 'epochs') and value is not None}
config['source_sha256'] = SOURCE_SHA
if (RUN / 'config.json').exists():
    assert json.loads((RUN / 'config.json').read_text()) == config, 'this run directory was started with other settings'
else:
    publish(RUN / 'config.json', lambda stream: json.dump(config, stream, indent=1))

torch.set_num_threads(1)
SPLIT = splits()
if args.train_file:
    SPLIT['train'] = [int(x) for x in Path(args.train_file).read_text().split(',')]
if args.val_file:
    SPLIT['val_a'] = [int(x) for x in Path(args.val_file).read_text().split(',')]
assert not set(SPLIT['train']) & set(SPLIT['val_a']) and not set(SPLIT['train']) & set(SPLIT['val_b'])
NETS, LABEL = len(SPLIT['train']), 'validation (%d held-out nets)' % len(SPLIT['val_a']) if args.val_file else 'VAL-A'
rng = np.random.default_rng(args.seed)
ORDER = [int(rank) for _ in range(args.epochs) for rank in rng.permutation(SPLIT['train'])]
BATCHES = [ORDER[i:i + args.batch] for i in range(0, len(ORDER) - args.batch + 1, args.batch)]
if args.smoke:
    BATCHES, SPLIT['val_a'] = BATCHES[:3], SPLIT['val_a'][:2]
TOTAL = len(BATCHES)

arrays, initial_dials = read_source(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
full = [p['head'], p['head_b'], e['output'], e['bias'], T['a'], T['c']]
full += [e[key] for key in e if key not in ('output', 'bias')]
trunk = [p[key] for key in TRUNK]
assert len(full + trunk) == 23 and sum(x.numel() for x in full) == 3437 and sum(x.numel() for x in trunk) == 44144
for value in full + trunk:
    value.requires_grad_(True)
TRAINABLE = [key for key, value in T.items() if value.requires_grad]
pairs = lambda text: {item.split(':')[0]: float(item.split(':')[1]) for item in text.split(',')} if text else {}
DIAL_RATE, DIAL_START = pairs(args.dial_lr), pairs(args.dial_init)
assert set(DIAL_RATE) <= set(DIAL_LR) and set(DIAL_START) <= set(DIAL_LR)
DL = {key: torch.tensor(value + DIAL_START.get(key, 0.0), dtype=torch.float64, requires_grad=key in DIAL_LR)
      for key, value in initial_dials.items()}
trained_dials = [DL[key] for key in DIAL_LR]
groups = [dict(params=full, base=args.lr_full), dict(params=trunk, base=args.lr_trunk)]
groups += [dict(params=[DL[key]], base=DIAL_RATE.get(key, value * args.dial_lr_scale)) for key, value in DIAL_LR.items()]
opt = torch.optim.Adam([dict(params=g['params'], lr=g['base']) for g in groups])
ema = {key: T[key].detach().to(torch.float64).clone() for key in TRAINABLE}
ema_dials = {key: DL[key].detach().clone() for key in DIAL_LR}
# Extra global physics dials (--extra): one shift per family, applied on every layer through the leaves of otr_pl.py; 0 = today.
EXTRA = {item.split(':')[0]: float(item.split(':')[1]) for item in args.extra.split(',')} if args.extra else {}
START = {item.split(':')[0]: float(item.split(':')[1]) for item in args.extra_init.split(',')} if args.extra_init else {}
assert set(START) <= set(EXTRA), 'a start value needs its dial in --extra'
XL = {name: torch.tensor(START.get(name, 0.0), dtype=torch.float64, requires_grad=True) for name in EXTRA}
ema_extra = {name: torch.tensor(START.get(name, 0.0), dtype=torch.float64) for name in EXTRA}
REFERENCE = bool(args.init) or any(START.values()) or any(DIAL_START.values())   # start point != untrained parent: score the parent once
extra_groups = [dict(params=[XL[name]], base=rate) for name, rate in EXTRA.items()]


def load_optimizer(saved):
    """Adam state of a checkpoint; one written before the extra dials existed gets fresh state for them."""
    if len(saved['param_groups']) == len(opt.param_groups):
        opt.load_state_dict(saved)
        for group in extra_groups:
            opt.add_param_group(dict(params=group['params'], lr=group['base']))
    else:
        for group in extra_groups:
            opt.add_param_group(dict(params=group['params'], lr=group['base']))
        opt.load_state_dict(saved)


def load_extras(state):
    global ema_extra
    with torch.no_grad():
        for name in XL:
            if name in state.get('extras', {}):
                XL[name].copy_(state['extras'][name])
                ema_extra[name] = state['ema_extras'][name].clone()


done = 0
OFFSET = 0   # updates already made by the run this one continues (--init); warm-up and EMA schedule count from there
if args.init:
    state = torch.load(args.init, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for key in T:
            T[key].copy_(state['arrays'][key])
        for key in DL:
            DL[key].copy_(state['dials'][key])
    load_optimizer(state['optimizer'])
    load_extras(state)
    ema, ema_dials = state['ema'], state['ema_dials']
    OFFSET = args.init_updates if args.init_updates is not None else state.get('total_updates', state['update'])
    del state
else:
    load_optimizer(opt.state_dict())
groups += extra_groups

log = (RUN / 'log.jsonl').open('a')
progress = (RUN / 'progress.txt').open('a')


def emit(text, **record):
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    line = json.dumps(dict(time=stamp, **record), allow_nan=False)
    log.write(line + '\n')
    log.flush()
    progress.write('%s %s\n' % (stamp, text))
    progress.flush()
    print(line, flush=True)


def note(text):
    emit(text, event='waiting', note=text)


def restore_frozen():
    for key in FROZEN_P:
        assert T['p.' + key].detach().numpy().tobytes() == arrays['p.' + key].tobytes()
    for key in FROZEN_DIALS:
        assert float(DL[key].detach()) == initial_dials[key] and not DL[key].requires_grad


def raw_parameters():
    return ({key: value.detach() for key, value in T.items()}, {key: float(value.detach()) for key, value in DL.items()},
            {name: [float(value.detach())] * 16 for name, value in XL.items()})


def ema_parameters():
    raw, dials, _ = raw_parameters()
    raw.update({key: value.to(torch.float32) for key, value in ema.items()})
    dials.update({key: float(value) for key, value in ema_dials.items()})
    return raw, dials, {name: [float(value)] * 16 for name, value in ema_extra.items()}


def keep(path, payload):
    """Publish a parameter file, or prove that the one already on disk (an earlier life of this run) is bit-identical."""
    if path.exists():
        old_arrays, old_dials = load_params(torch, path)
        assert all(torch.equal(old_arrays[key], payload[0][key]) for key in payload[0]) and old_dials == payload[1] \
            and load_pl(torch, path) == payload[2], '%s on disk differs from the replayed trajectory' % path
    else:
        save_params(torch, path, *payload)
    return str(path)


def evaluate(k):
    """Forward-only final-layer MSE of the validation networks for the raw and the EMA parameters after k updates.
    A run that continues another one (--init) also scores the untrained parameters once, as the reference."""
    files = {}
    if REFERENCE and k == 0:
        files['untrained'] = keep(RUN / 'params' / 'untrained.pt', (
            {key: torch.from_numpy(value) for key, value in arrays.items()}, dict(initial_dials), {}))
    if args.eval_what != 'ema' or (k == 0 and not REFERENCE):
        files['raw'] = keep(RUN / 'params' / ('p%05d.pt' % k), raw_parameters())
    if k or args.init:
        files['ema'] = keep(RUN / 'params' / ('ema%05d.pt' % k), ema_parameters())
    requests = {name: [dict(id='e%05d_%s_r%04d' % (k, name, rank), mode='eval', rank=rank, params=path)
                       for rank in SPLIT['val_a']] for name, path in files.items()}
    started = time.monotonic()
    for request in sum(requests.values(), []):
        enqueue(RUN, request)
    wait_results(RUN, sum(requests.values(), []), note)
    record = dict(event='eval', update=k, ranks=SPLIT['val_a'], seconds=time.monotonic() - started)
    for name, items in requests.items():
        rows = [json.loads(result_path(RUN, request).read_text()) for request in items]
        assert all(row.get('card', 'L') == (args.card or 'L') for row in rows), 'a result came from a worker on another card'
        record[name] = [row['mse'] for row in rows]
        record[name + '_mean'] = float(np.mean(record[name]))
        record[name + '_all_layers_mean'] = float(np.mean([row['mse_layers'] for row in rows]))
    first = record if k == 0 else next(json.loads(line) for line in (RUN / 'log.jsonl').read_text().splitlines()
                                       if '"event": "eval"' in line and json.loads(line)['update'] == 0)
    baseline = first['untrained'] if 'untrained' in first else first['raw']
    text = '%s after %d updates:' % (LABEL, k + OFFSET)
    if list(files) == ['untrained']:
        text += ' untrained reference %.4e (all layers %.4e)' % (record['untrained_mean'], record['untrained_all_layers_mean'])
    for name in files:
        if name == 'untrained':
            continue
        record[name + '_vs_untrained_pct'] = 100 * (record[name + '_mean'] / float(np.mean(baseline)) - 1)
        record[name + '_wins'] = int(sum(a < b for a, b in zip(record[name], baseline)))
        text += ' %s %.4e (%+.2f%% vs untrained, better on %d/%d)' % (
            name, record[name + '_mean'], record[name + '_vs_untrained_pct'], record[name + '_wins'], len(baseline))
    emit(text, **record)


def checkpoint(k):
    path = RUN / 'ckpt' / ('after_%05d.pt' % k)
    if path.exists():
        return
    state = dict(update=k, total_updates=k + OFFSET, extras={name: value.detach().clone() for name, value in XL.items()},
                 ema_extras={name: value.clone() for name, value in ema_extra.items()},
                 config=config, arrays={key: value.detach().clone() for key, value in T.items()},
                 dials={key: value.detach().clone() for key, value in DL.items()},
                 ema={key: value.clone() for key, value in ema.items()},
                 ema_dials={key: value.clone() for key, value in ema_dials.items()}, optimizer=opt.state_dict())
    publish(path, lambda stream: torch.save(state, stream), 'wb')


saved = sorted((RUN / 'ckpt').glob('after_*.pt'))
if saved:
    state = torch.load(saved[-1], map_location='cpu', weights_only=True)
    assert state['config'] == config and set(state['arrays']) == set(T)
    with torch.no_grad():
        for key in T:
            T[key].copy_(state['arrays'][key])
        for key in DL:
            DL[key].copy_(state['dials'][key])
    opt.load_state_dict(state['optimizer'])
    load_extras(state)
    ema, ema_dials, done = state['ema'], state['ema_dials'], state['update']
    del state
restore_frozen()
emit('%s at update %d of %d (%d epochs of %d nets, batch %d; %d earlier updates behind --init); settings %s' % (
    'RESUME' if saved else 'START', done, TOTAL, args.epochs, NETS, args.batch, OFFSET, json.dumps(config)),
    event='resume' if saved else 'start', done=done, total=TOTAL, epochs=args.epochs, config=config)
if done == 0:
    evaluate(0)

while done < TOTAL:
    batch = BATCHES[done]
    path = keep(RUN / 'params' / ('p%05d.pt' % done), raw_parameters())
    attach = (RUN / 'STAR_ATTACH').exists() and (RUN / 'STAR_ATTACH').read_text().strip() == '1'
    requests = [dict(id='g%05d_%02d_r%04d' % (done, i, rank), mode='grad', rank=rank, params=path, scale=1e8 / len(batch),
                     **(dict(attach=1) if attach else {})) for i, rank in enumerate(batch)]
    started = time.monotonic()
    replayed = all(result_path(RUN, request).exists() for request in requests)
    for request in requests:
        enqueue(RUN, request)
    wait_results(RUN, requests, note)
    rows = [torch.load(result_path(RUN, request), map_location='cpu', weights_only=True) for request in requests]
    assert all(row['params'] == path and row['rank'] == rank and row['scale'] == 1e8 / len(batch)
               for row, rank in zip(rows, batch))
    assert all(row.get('card', 'L') == (args.card or 'L') for row in rows), 'a result came from a worker on another card'
    for pg, group in zip(opt.param_groups, groups):
        pg['lr'] = group['base'] * min(1., (done + OFFSET + 1) / args.warm)
    opt.zero_grad(set_to_none=True)
    for key in TRAINABLE:
        total = rows[0]['grads'][key].clone()
        for row in rows[1:]:
            total += row['grads'][key]
        T[key].grad = total
    for key in DIAL_LR:
        total = rows[0]['dial_grads'][key].clone()
        for row in rows[1:]:
            total += row['dial_grads'][key]
        DL[key].grad = total
    assert not XL or all('pl_grads' in row for row in rows), '--extra needs workers started with TRAINER_PL=1'
    for name in XL:      # a global dial feels the sum of its per-layer gradients
        XL[name].grad = sum(row['pl_grads'][name].sum() for row in rows).clone()
    norm = float(torch.nn.utils.clip_grad_norm_(full + trunk, args.clip, error_if_nonfinite=True))
    dial_norm = float(torch.nn.utils.clip_grad_norm_(trained_dials, args.clip, error_if_nonfinite=True))
    extra_norm = float(torch.nn.utils.clip_grad_norm_(list(XL.values()), args.clip, error_if_nonfinite=True)) if XL else 0.0
    opt.step()
    restore_frozen()
    assert all(torch.isfinite(value).all() for value in list(T.values()) + list(DL.values()))
    done += 1
    decay = min(args.ema, (1 + done + OFFSET) / (10 + done + OFFSET))
    with torch.no_grad():
        for key in TRAINABLE:
            ema[key].mul_(decay).add_(T[key].to(torch.float64), alpha=1 - decay)
        for key in DIAL_LR:
            ema_dials[key].mul_(decay).add_(DL[key], alpha=1 - decay)
        for name in XL:
            ema_extra[name].mul_(decay).add_(XL[name], alpha=1 - decay)
    mse = [row['mse'] for row in rows]
    seconds = time.monotonic() - started
    emit('update %d/%d epoch %d  batch-mean final-layer MSE %.4e  grad norm %.2f dials %.2f  %.0f s%s%s' % (
        done, TOTAL, (done - 1) * args.batch // NETS, float(np.mean(mse)), norm, dial_norm, seconds, ' (replayed)' * replayed,
        ''.join('  %s %+.4f' % (name, float(value.detach())) for name, value in XL.items())),
        event='update', update=done, ranks=batch, mse=mse, mse_mean=float(np.mean(mse)),
        mse_layers_mean=np.mean([row['mse_layers'] for row in rows], axis=0).tolist(),
        lr=[pg['lr'] for pg in opt.param_groups[:2]], grad_norm=norm, dial_grad_norm=dial_norm, ema_decay=decay,
        dials={key: float(value.detach()) for key, value in DL.items()},
        extras={name: float(value.detach()) for name, value in XL.items()}, extra_grad_norm=extra_norm,
        edge_scalars={key: float(e[key].detach()) for key in ('cov_factor', 'rho_scale', 'third_scale')},
        seconds=seconds, replayed=replayed, star_attach=sorted({bool(row.get('star_attach', False)) for row in rows}),
        net_seconds=[row['seconds'] for row in rows], workers=sorted({row['worker'] for row in rows}))
    if done % args.ckpt_every == 0 or done == TOTAL:
        checkpoint(done)
    if done % args.eval_every == 0 or done == TOTAL:
        evaluate(done)
emit('DONE: %d updates' % done, event='done', update=done)
