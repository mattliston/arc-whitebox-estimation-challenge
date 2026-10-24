"""TAG fable-lead 2026-10-19: ROUND-2 dial arms (zero bill) built on the v3 micro-stack physics (science base + mstack3 dial moves,
candidate_ms3base_sci_v1.py): per-COMPONENT splits of the vector dials whose blanket moves showed signal: _HC_U rows (5 hidden
columns) x0.8/x1.2, _NP_STRENGTH and _NP_RHO by band (L8-11, L12-14), _K4L34_BY_LAYER by band (0.3/0.7), _NP_K3L_BY_LAYER by age
(x0.9/x1.1), _K31R_TABLE row factor early (L1-7) / late (L8-14) x0.9/x1.1. Also writes the reference arm ms3base."""
import ast
import re

base_lines = open('candidate_stk7_sci_v1.py').read().split('\n')
assert base_lines[-2].startswith('_FAST_RESCUE_CAP_S')
moves = [l for l in open('mstack3_dials_v1.txt').read().split('\n') if l.strip() and not l.startswith('#')]
ms3 = base_lines[:-2] + ['# [fable-lead 2026-10-19] v3 micro-stack physics (mstack3 dial moves) as the ROUND-2 science base'] + moves + base_lines[-2:]
ms3_text = '\n'.join(ms3); compile(ms3_text, 'ms3', 'exec'); open('candidate_ms3base_sci_v1.py', 'w').write(ms3_text)
L = ms3_text.split('\n')


def lit(name):
    v = [l for l in L if l.startswith(name + ' = ')][-1][len(name) + 3:].split('#')[0].strip()
    return ast.literal_eval(v)


def write(tag, assigns, note):
    out = L[:-2] + ['# [fable-lead 2026-10-19] ROUND-2 arm %s on the v3 physics: %s' % (tag, note)] + assigns + L[-2:]
    t = '\n'.join(out); compile(t, tag, 'exec'); open('candidate_%s_sci_v1.py' % tag, 'w').write(t); print(tag)


tags = []
hc = lit('_HC_U')
for r in range(5):
    for s, sfx in ((0.8, 'm'), (1.2, 'p')):
        new = tuple(tuple(x * s for x in row) if i == r else row for i, row in enumerate(hc))
        tag = 'hcu_r%d_%s' % (r, sfx); write(tag, ['_HC_U = ' + repr(new)], 'hidden column %d (%s) weights x%s' % (r, ('diag', 'k4_local_rows', 'm2', 'eold', 'qb')[r], s)); tags.append(tag)
nps = lit('_NP_STRENGTH'); rho = lit('_NP_RHO')
for band, layers in (('b2', range(8, 12)), ('b3', range(12, 15))):
    for s, sfx in ((0.9, 'm'), (1.1, 'p')):
        tag = 'npstr_%s_%s' % (band, sfx); write(tag, ['_NP_STRENGTH = ' + repr(tuple(x * s if i in layers else x for i, x in enumerate(nps)))], 'pair-law strengths at layers %s x%s' % (list(layers), s)); tags.append(tag)
    for s, sfx in ((0.95, 'm'), (1.05, 'p')):
        tag = 'nprho_%s_%s' % (band, sfx); write(tag, ['_NP_RHO = ' + repr(tuple(x * s if i in layers else x for i, x in enumerate(rho)))], 'NP rho at layers %s x%s' % (list(layers), s)); tags.append(tag)
k4l34 = lit('_K4L34_BY_LAYER')
for band, layers in (('b2', range(8, 12)), ('b3', range(12, 16))):
    for s, sfx in ((0.3, 'm'), (0.7, 'p')):
        tag = 'k4l34_%s_%s' % (band, sfx); write(tag, ['_K4L34_BY_LAYER = ' + repr({l: ((s, s) if l in layers else v) for l, v in k4l34.items()})], 'K4L ages 3-4 strengths %s at layers %s' % (s, list(layers))); tags.append(tag)
k3l = lit('_NP_K3L_BY_LAYER')
for a in range(3):
    for s, sfx in ((0.9, 'm'), (1.1, 'p')):
        tag = 'k3l_a%d_%s' % (a + 1, sfx); write(tag, ['_NP_K3L_BY_LAYER = ' + repr({l: tuple(x * s if i == a else x for i, x in enumerate(v)) for l, v in k3l.items()})], 'K3L age-%d strengths (L7-14) x%s' % (a + 1, s)); tags.append(tag)
m = re.search(r'^_K31R_TABLE = \{l: \((\S+), (\S+)\) for l in _K31R_TABLE\}', ms3_text, re.M); a0, b0 = float(m.group(1)), float(m.group(2))
for band, cond in (('early', 'l <= 7'), ('late', 'l >= 8')):
    for s, sfx in ((0.9, 'm'), (1.1, 'p')):
        tag = 'k31r_%s_%s' % (band, sfx); write(tag, ['_K31R_TABLE = {l: ((%r, %r) if %s else (%r, %r)) for l in _K31R_TABLE}' % (a0 * s, b0, cond, a0, b0)], 'K31R row factor x%s at %s' % (s, cond)); tags.append(tag)
open('round2_tags_v1.txt', 'w').write('\n'.join(['ms3base'] + tags) + '\n'); print(len(tags) + 1, 'arms')
