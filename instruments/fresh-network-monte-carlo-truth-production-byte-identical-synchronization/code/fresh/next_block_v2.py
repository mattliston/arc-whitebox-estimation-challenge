#!/usr/bin/env python3
"""TAG analyst truth order by need v2 (lead 2026-10-12 11:2xZ: v1 extended to S64-S66 = A114-A131, the trainer's maps of 11:14:50Z; nothing
else changed). v1 text: which truth block to produce next. Block names, seeds and sample counts stay as in the
chain's numbering and the trainer's gen_round maps (block Ab = seeds 7100712 + 64 (b - 43) ... + 63; 16M when (b - 43) is a multiple of 3, else 8M);
only the ORDER changes. Per round, by need: phase-1 blocks (1 and 3), then the check / sealed block (2), then the reserve (5); phase-2 blocks (4 and
6) only while the essentials of at least the next TWO rounds after the trainer's current round exist.
  S54-S66 below (v2; v1 stopped at S63). S52 (own map): A46 sizing, A49 check / sealed (A43-A45 done before this order). S53 (own map): A48 + A50 phase 1, A52 check / sealed, A47 reserve;
  phase 2 A51 + A53. S54-S63 (gen_round v1): base = 54 + 6 (R - 54): phase 1 base and base + 2, check / sealed base + 1, reserve base + 4; phase 2
  base + 3 and base + 5.
The trainer's current round T = the first round from S51 whose O/trainer/run_RAD_S<R>_gen/STOP does not exist. Rounds before T are finished: their
missing blocks are skipped. A block exists when all 64 truth files O/analyst/fresh/truth_seed<seed>.npz exist.
usage: next_block_v1.py next                      -> 'b k1 k2 half_batches round role' for the next block, or 'NONE'
       next_block_v1.py plan <start ISO> <T> <next round start ISO> <round hours>   -> the order with ETAs (16M 2.33 h, 8M 1.17 h a block)"""
import sys, datetime as dt
from pathlib import Path
F = Path(__file__).resolve().parent; O = F.parents[1]
ESS = {52: [46, 49], 53: [48, 50, 52, 47]}; PH2 = {52: [], 53: [51, 53]}
ROLE = {46: 'sizing', 49: 'check / sealed', 48: 'phase 1', 50: 'phase 1', 52: 'check / sealed', 47: 'reserve', 51: 'phase 2', 53: 'phase 2'}
for R in range(54, 67):
    b0 = 54 + 6 * (R - 54); ESS[R] = [b0, b0 + 2, b0 + 1, b0 + 4]; PH2[R] = [b0 + 3, b0 + 5]
    ROLE.update({b0: 'phase 1', b0 + 2: 'phase 1', b0 + 1: 'check / sealed', b0 + 4: 'reserve', b0 + 3: 'phase 2', b0 + 5: 'phase 2'})
ROUND = {b: R for R in ESS for b in ESS[R] + PH2[R]}


def blk(b):
    k1 = 712 + 64 * (b - 43); return k1, k1 + 63, (244 if (b - 43) % 3 == 0 else 122)


def exists(b):
    k1, k2, _ = blk(b); return all((F / f'truth_seed{7100000 + k}.npz').exists() for k in range(k1, k2 + 1))


def current_round():
    for R in range(51, 67):
        if not (O / f'trainer/run_RAD_S{R}_gen/STOP').exists(): return R
    return 67


def choose(have, T):
    k = 0
    for R in range(T + 1, 67):
        if all(have(b) for b in ESS[R]): k += 1
        else: break
    if k >= 2:
        for R in range(T + 1, T + k + 1):
            for b in PH2[R]:
                if not have(b): return b
    for R in range(max(52, T), 67):
        for b in ESS[R]:
            if not have(b): return b
    for R in range(max(52, T + 1), 67):
        for b in PH2[R]:
            if not have(b): return b
    return None


if sys.argv[1] == 'next':
    b = choose(exists, current_round())
    print('NONE' if b is None else ' '.join(map(str, (b, *blk(b), ROUND[b], ROLE[b].replace(' ', '_')))))
else:
    t = dt.datetime.fromisoformat(sys.argv[2].replace('Z', '+00:00')); T0 = int(sys.argv[3]); t1 = dt.datetime.fromisoformat(sys.argv[4].replace('Z', '+00:00'))
    h = float(sys.argv[5]); made = {b for b in range(43, 132) if exists(b)}
    T = lambda now: T0 if now < t1 else T0 + 1 + int((now - t1).total_seconds() // (3600 * h))
    while True:
        b = choose(lambda x: x in made, T(t))
        if b is None: break
        t += dt.timedelta(hours=2.33 if blk(b)[2] == 244 else 1.17); made.add(b)
        print(f"A{b} S{ROUND[b]} {ROLE[b]} {7100000 + blk(b)[0]}-{7100000 + blk(b)[1]} {'16M' if blk(b)[2] == 244 else '8M'} ready {t:%m-%d %H:%MZ} (trainer then in S{T(t)})")
