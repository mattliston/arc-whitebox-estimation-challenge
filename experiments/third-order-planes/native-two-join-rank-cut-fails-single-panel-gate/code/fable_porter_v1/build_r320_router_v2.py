#!/usr/bin/env python3
"""TAG fable-porter (PT-60d: versioned copy of build_r320_router.py; the ONLY changes: a fixed per-join rank may be any multiple of 32 in
[64, 512] (lead 09:3x: the analyst may pick 256 at L13 or 448 at L7), and when the largest fixed rank exceeds 384 the rotation overflow
envelope's width literal 384.0 becomes that rank (it must bound every rank). Threshold rules still pick 320 or 384. Original: TAG fable-porter (PT-60, lead 09:0x "prepare the native per-join R320 router"): the old-tier frame rank becomes a PER-JOIN runtime choice.
Count-asserted edits of the LAST definitions in a pre-treuse candidate (treuse is appended after this, LAST, as always):
  _predict_binary     clears _R320_RESP at entry; records resp_L = len(response_indices) (the card's second select, the analyst's resp_L) at
                      every layer right after the response selection, i.e. BEFORE that layer's push/join (causal for a same-layer join)
  _shared_old384_push rank = _r320_rank(layer) at a join (else the current frame rank, state['rank']); the live-row guard uses that rank; the new
                      frame rank is staged as state['rank_new'] after the QR shape check and becomes state['rank'] before _slr_finish_factors
  _slr_layout / _slr_join_factors / _slr_reconstruct   shape asserts use the TRACKED ranks (state['rank'], the lazy group's base_rank/rank)
                      instead of the fixed _SLR_RANK = 384, so mixed per-join ranks (e.g. 384 at L7, 320 at L10/L13) pass the same checks
+ appended fragment: _R320_RULE (from <rule.json>: {join layer: 320 | 384 | [feature layer, threshold]}), _R320_RESP, _r320_rank(layer)
  (a threshold rule reads resp at the feature layer; it raises if that layer has not run yet: causality guard).
The overflow envelope's width literal 384.0 stays (a bound for any rank <= 384). A rule of all 384 must reproduce the input bit for bit; all 320
must reproduce build_tier_rank.py <in> 320. usage: build_r320_router.py <in.py> <out.py> <rule.json>"""
import sys, json, hashlib
from pathlib import Path
src_path, out_path, rule_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
assert not out_path.exists(), f'refusing to overwrite {out_path}'
rule = {int(k): (v if isinstance(v, int) else (int(v[0]), int(v[1]))) for k, v in json.loads(rule_path.read_text()).items()}
assert all(isinstance(v, int) and v % 32 == 0 and 64 <= v <= 512 or isinstance(v, tuple) and v[0] <= k for k, v in rule.items()), ('rule: fixed multiple of 32 in [64, 512] or [feature layer <= join layer, threshold]', rule)
RMAX = max([v for v in rule.values() if isinstance(v, int)] + [384])
src = src_path.read_text()
def last_fn(text, name):
    i = text.rfind(f'\ndef {name}('); assert i >= 0, name
    j = text.find('\ndef ', i + 5); j = len(text) if j < 0 else j
    return i, j
def edit_fn(text, name, pairs):
    i, j = last_fn(text, name); fn = text[i:j]
    for old, new in pairs:
        assert fn.count(old) == 1, (name, fn.count(old), old[:80]); fn = fn.replace(old, new)
    return text[:i] + fn + text[j:]
# _predict_binary: reset at entry, record resp_L per layer
i, j = last_fn(src, '_predict_binary'); fn = src[i:j]; k = fn.index(':\n', fn.index('def _predict_binary(')) + 2
fn = fn[:k] + '    _R320_RESP.clear()   # PT-60 R320 router: per-MLP runtime features\n' + fn[k:]
a = "            response_indices, response_mask = select(response_score, gate_floor)\n"
assert fn.count(a) == 1, fn.count(a)
fn = fn.replace(a, a + "            _R320_RESP[l] = int(response_indices.shape[0])   # PT-60: resp_L, before this layer's push/join\n")
src = src[:i] + fn + src[j:]
# live push: per-join rank, guard, staged frame rank
src = edit_fn(src, '_shared_old384_push', [
    ("    if int(w32.shape[0]) != 1024 or int(previous.shape[0]) < 384:\n",
     "    rank = _r320_rank(layer) if joins_here is not None else (int(state['rank']) if state.get('rank') else 384)   # PT-60 per-join rank\n"
     "    if int(w32.shape[0]) != 1024 or int(previous.shape[0]) < rank:\n"),
    ("    rank = 384\n", "    # PT-60: rank chosen above (per join)\n"),
    ("        if tuple(Qn.shape) != (int(previous.shape[0]), rank):\n            raise RuntimeError('Shared-old384 QR shape')\n",
     "        if tuple(Qn.shape) != (int(previous.shape[0]), rank):\n            raise RuntimeError('Shared-old384 QR shape')\n        state['rank_new'] = rank   # PT-60: the new frame's rank\n"),
    ("        _slr_finish_factors(state, factors)\n", "        state['rank'] = state.pop('rank_new')   # PT-60\n        _slr_finish_factors(state, factors)\n")])
# lazy rotation: tracked ranks instead of _SLR_RANK
src = edit_fn(src, '_slr_layout', [
    ("_slr_matrix(state['F'], _SLR_RANK, columns, 'materialized factors')", "_slr_matrix(state['F'], int(state['rank']), columns, 'materialized factors')"),
    ("_slr_matrix(group['base'], _SLR_RANK, old, 'unrotated prefix')", "_slr_matrix(group['base'], int(group['base_rank']), old, 'unrotated prefix')"),
    ("_slr_matrix(group['join'], _SLR_RANK, joined, 'new cohort factors')", "_slr_matrix(group['join'], int(group['rank']), joined, 'new cohort factors')"),
    ("_slr_matrix(group['map'], _SLR_RANK, _SLR_RANK, 'full-rank coordinate map')", "_slr_matrix(group['map'], int(group['rank']), int(group['base_rank']), 'coordinate map (tracked ranks)')")])
src = edit_fn(src, '_slr_join_factors', [
    ("_slr_matrix(overlap, _SLR_RANK, _SLR_RANK, 'current overlap')", "_slr_matrix(overlap, int(state['rank_new']), int(state['rank']), 'current overlap (tracked ranks)')"),
    ("int(_e2_H(Fj.shape[0])) == _SLR_RANK", "int(_e2_H(Fj.shape[0])) == int(state['rank_new'])"),
    ("state['lazy'] = dict(layer=layer, prefix=old, joined=int(_e2_H(Fj.shape[1])), base=state['F'], join=Fj, map=overlap)",
     "state['lazy'] = dict(layer=layer, prefix=old, joined=int(_e2_H(Fj.shape[1])), base=state['F'], join=Fj, map=overlap, base_rank=int(state['rank']), rank=int(state['rank_new']))")])
src = edit_fn(src, '_slr_reconstruct', [("(int(_e2_H(U.shape[1])) == _SLR_RANK)", "(int(_e2_H(U.shape[1])) == int(state['rank']))")])
if RMAX > 384:
    src = edit_fn(src, '_slr_virtual_rotation_envelope', [("width = fnp.full((1, 1), 384.0, dtype=fnp.float64)", f"width = fnp.full((1, 1), {float(RMAX)!r}, dtype=fnp.float64)   # PT-60d: bound for ranks <= {RMAX}")])
frag = ('\n\n# BEGIN FABLE-PORTER PER-JOIN R320 ROUTER (PT-60, 2026-09-25; build_r320_router.py; rule: ' + json.dumps({str(k): (v if isinstance(v, int) else list(v)) for k, v in rule.items()}) + ')\n'
        f'_R320_RULE = {rule!r}   # join layer -> 320 | 384 | (feature layer, threshold): 320 iff resp at that layer >= threshold\n'
        '_R320_RESP = {}\n\n\n'
        'def _r320_rank(layer):\n'
        '    r = _R320_RULE[layer]\n'
        '    if isinstance(r, int):\n'
        '        return r\n'
        '    L, threshold = r\n'
        '    if L not in _R320_RESP:\n'
        "        raise RuntimeError('R320 router: resp at L%d not computed before the L%d join' % (L, layer))\n"
        '    return 320 if _R320_RESP[L] >= threshold else 384\n'
        '# END FABLE-PORTER PER-JOIN R320 ROUTER\n')
out = src.rstrip('\n') + '\n' + frag
compile(out, str(out_path), 'exec')
out_path.write_text(out)
print(dict(out=str(out_path), rule=rule, sha256=hashlib.sha256(out.encode()).hexdigest()[:16]))
