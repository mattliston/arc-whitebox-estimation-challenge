"""TAG opus-trainer RAD (10-12 08:0xZ; lead 07:5xZ: "after S54, rounds keep this template until 10-17; the final round's freeze no later than
10-17 12:00Z"): writes round S<N> (N >= 55) from S54's files, one round = the next 6 analyst blocks (S54 = A54-A59, S55 = A60-A65, ...):
sizing phase 1 = blocks 1 + 3, phase 2 = blocks 4 + 6 (8M), check = block 2's first half, sealed = block 2's second half (16M, the physicist),
reserve = block 5 (16M). Every seed of S54's files moves by 384 x (N - 54), every block name by 6 x (N - 54); the round lists of make_s54.py
run back to S50 / S51; S54 -> S<N> and S53 -> S<N-1> (release tags s53* -> s<N-1>*). The chain gets the lead's deadline: phase 2 only if the
projected freeze also lands before 10-17 12:00Z, and the round does not start after 10-17 00:00Z.
Writes (all new files; refuses to overwrite): cert/RAD/S<N>/{make_s<N>.py, hold_s<N>.sh, chain_s<N>.sh, PORT_RULE_v1.json},
cert/RAD/gen_allow_v<10 + N - 54>.txt, cert/RAD/launch_gen_v<10 + N - 54>.sh and _t2.sh. Checks: no sealed / reserve seed in the allow list.
usage (from O/trainer): python -B cert/RAD/gen_round.py <N>"""
import json
import os
import re
import sys
from pathlib import Path

N = int(sys.argv[1]); assert N >= 55
P, K, SHIFT, OFF = N - 1, 10 + N - 54, 384 * (N - 54), 6 * (N - 54)
first = lambda b: 7100000 + 712 + 64 * (b - 43)
B = [54 + OFF + i for i in range(6)]      # blocks 1..6 of this round
assert all(((b - 46) % 3 == 0) == (i in (1, 4)) for i, b in enumerate(B)), 'blocks 2 / 5 must be the 16M ones'
HOME = Path('cert/RAD/S%d' % N)
utc = os.popen('date -u +%FT%TZ').read().strip()


def shift_text(s):
    s = re.sub(r'(?<!\d)(710(?:14|15|16|17)\d\d)-(\d{3})(?!\d)',      # S54's short ranges ('7101416-479') -> full shifted ranges
               lambda m: '%d-%d' % (int(m.group(1)) + SHIFT, int(m.group(1)[:-3] + m.group(2)) + SHIFT), s)
    s = re.sub(r'(?<!\d)710(?:14|15|16|17)\d\d(?!\d)', lambda m: str(int(m.group(0)) + SHIFT), s)      # also inside ids ('_r7101480')
    s = re.sub(r'\bA(5[4-9])\b', lambda m: 'A%d' % (int(m.group(1)) + OFF), s)
    return s


def rounds(s):
    keep = {      # protected first (the generic S54 / S53 renaming must not touch them), filled last
        "('S53', 'S52', 'S51', 'S50')": '(%s)' % ', '.join("'S%d'" % r for r in range(P, 49, -1)),
        "('S51', 'S52', 'S53')": '(%s)' % ', '.join("'S%d'" % r for r in range(51, N)),
        "('S53', 'S52', 'S51')": '(%s)' % ', '.join("'S%d'" % r for r in range(P, 50, -1)),
        "(S53's solve, else S52's, else S51's)": "(the latest round with a solve, S%d back to S51)" % P,
        "S53's forward arms with single-arm LOO < 0 on S53's sizing networks (else S52's, else S51's)":
            "the forward arms of the latest round with a solve (S%d back to S51) with single-arm LOO < 0 on its sizing networks" % P,
        'S53 / S52 / S51 / S50': 'S%d back to S50' % P,
        'make_s53_v3.py': 'make_s53_v3.py', 'chain_s53_v3.sh': 'chain_s53_v3.sh'}
    toks = {}
    for i, (old, new) in enumerate(keep.items()):
        toks['\x01L%d\x01' % i] = new
        s = s.replace(old, '\x01L%d\x01' % i)
    s = s.replace('S54', '\0N').replace('s54', '\0n').replace('S53', '\0P').replace('s53', '\0p')
    s = s.replace('\0N', 'S%d' % N).replace('\0n', 's%d' % N).replace('\0P', 'S%d' % P).replace('\0p', 's%d' % P)
    for tok, new in toks.items():
        s = s.replace(tok, new)
    assert '\0' not in s and '\x01' not in s
    return s


def write(path, text):
    path = Path(path); assert not path.exists(), path
    path.write_text(text)


HOME.mkdir(exist_ok=False)
tag = 'gen_round.py %d (lead 07:5xZ: rounds keep the S54 template until 10-17), written %s' % (N, utc)
m = rounds(shift_text(Path('cert/RAD/S54/make_s54.py').read_text()))
m = m.replace('(lead 10-12 07:5xZ: APPROVED as proposed; = make_s53_v3.py\'s template and routes, one round later)', '(%s)' % tag)
m = m.replace("note='trainer RAD S%d (lead 10-12 07:5xZ)" % N, "note='trainer RAD S%d (gen_round.py)" % N)
write(HOME / ('make_s%d.py' % N), m)
write(HOME / ('hold_s%d.sh' % N), rounds(Path('cert/RAD/S54/hold_s54.sh').read_text()))
c = rounds(shift_text(Path('cert/RAD/S54/chain_s54.sh').read_text()))
c = c.replace('launch_gen_v10', 'launch_gen_v%d' % K).replace('gen_allow_v10', 'gen_allow_v%d' % K)
c = c.replace("(lead 10-12 08:0xZ: APPROVED as proposed; = S%d's chain_s53_v3.sh one round later)" % P, '(%s, from chain_s54.sh)' % tag)
old_wait = "T_START=$(date +%s)\n"
assert c.count(old_wait) == 1
c = c.replace(old_wait, 'T_START=$(date +%s); LATE=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "2026-10-17T00:00:00Z" +%s); '
                        'DEAD=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "2026-10-17T12:00:00Z" +%s)\n'
                        '[ $T_START -lt $LATE ] || { log "S' + str(N) + ' NOT STARTED: past 10-17 00:00Z (lead: the final round freezes by 10-17 12:00Z)"; exit 0; }\n')
old_rule = 'if [ $PROJ -lt $(( T_START + 43200 )) ]; then'
assert c.count(old_rule) == 1
c = c.replace(old_rule, 'if [ $PROJ -lt $(( T_START + 43200 )) ] && [ $PROJ -lt $DEAD ]; then      # gen_round: + the 10-17 12:00Z freeze deadline')
write(HOME / ('chain_s%d.sh' % N), c)
write(HOME / 'PORT_RULE_v1.json', json.dumps(dict(
    utc=utc, round='S%d' % N, written_before_any_arm_result_was_read=True,
    rule="an arm whose native parity is FAILED before the freeze is excluded from the freeze; an arm with no verdict by the freeze is kept "
         "(the lead's S51 rule of 07:2xZ, data-independent; rounds keep the S54 template, lead 07:5xZ)",
    failed_line="the engineer's TEAM_LOG line '[engineer] <UTC stamp> S%d PORT PARITY <arm>: FAILED' (fixed string, lane-anchored; <arm> as in "
                "S%d's keep_v1.json)" % (N, N),
    applied_by='cert/RAD/S%d/chain_s%d.sh at the freeze: keep_v2.json = keep_v1.json minus the FAILED arms (keep_v1.json stays as written)' % (N, N)), indent=1))
seeds = list(range(7100264, 7100328))
for b in (B[0], B[2], B[3], B[5]):
    seeds += list(range(first(b), first(b) + 64))
seeds += list(range(first(B[1]), first(B[1]) + 32))
banned = set(range(7100168, 7100232)) | set(range(7100552, 7100584)) | set(range(7100680, 7100712)) | set(range(7100968, 7101032))
for b in range(46, B[5] + 1):
    if (b - 46) % 3 == 0:      # every 16M block: its sealed second half, and the reserve ones whole
        banned |= set(range(first(b) + 32, first(b) + 64))
for b in range(58, B[5] + 1, 6):
    banned |= set(range(first(b), first(b) + 64))
assert len(seeds) == len(set(seeds)) == 352 and not banned & set(seeds)
write('cert/RAD/gen_allow_v%d.txt' % K, ','.join(map(str, sorted(seeds))))
for src, dst in (('launch_gen_v10.sh', 'launch_gen_v%d.sh' % K), ('launch_gen_v10_t2.sh', 'launch_gen_v%d_t2.sh' % K)):
    t = Path('cert/RAD/' + src).read_text().replace('launch_gen_v10', 'launch_gen_v%d' % K).replace('gen_allow_v10', 'gen_allow_v%d' % K)
    t = t.replace('(10-12 07:5xZ, S54)', '(%s, S%d)' % (utc, N))
    t = t.replace('(A36 + A54 + A56 + A57 + A59 + A55 check 7101480-511; sealed sets and the A47 / A58 reserves NOT allowed)',
                  '(A36 + A%d + A%d + A%d + A%d + A%d check %d-%d; sealed sets and the reserves NOT allowed)'
                  % (B[0], B[2], B[3], B[5], B[1], first(B[1]), first(B[1]) + 31))
    write('cert/RAD/' + dst, t)
print('S%d written: blocks A%d-A%d (sizing A%d + A%d / A%d + A%d, check A%d %d-%d, sealed %d-%d, reserve A%d), allow v%d, launchers v%d'
      % (N, B[0], B[5], B[0], B[2], B[3], B[5], B[1], first(B[1]), first(B[1]) + 31, first(B[1]) + 32, first(B[1]) + 63, B[4], K, K))
