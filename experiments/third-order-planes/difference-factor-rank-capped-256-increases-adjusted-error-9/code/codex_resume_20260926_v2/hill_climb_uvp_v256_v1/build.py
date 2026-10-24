"""One fixed local V256 representation, using stdlib source transformations only."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
H = D.parent
PARENT = H / 'hill_climb_pair_rr_inverse_v1/candidate.py'
PARENT_SHA = 'a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'


def pin(path):
    b = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def write(name, value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2) + '\n'
    with (D / name).open('x') as f:
        f.write(text)


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    source = PARENT.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits = []

    def change(name, marker, pairs):
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        bodies = [''.join(lines[n.lineno - 1:n.end_lineno]) for n in nodes]
        selected = [b for b in bodies if marker in b]
        assert len(selected) == 1, (name, marker)
        before = after = selected[0]
        for old, new in pairs:
            after = replace_once(after, old, new)
        assert before != after
        edits.append(dict(function=name, before=before, after=after))

    change('_uvp_split', 'Fv2 = fnp.empty', [
        ('F_v/2 rank x P', 'F_v/2 leading min(256,rank) rows x P; original rank metadata'),
        ('    ru = min(_UVP_RU, rank)\n', '    ru = min(_UVP_RU, rank)\n    rv = min(256, rank)\n'),
        ('Fv2 = fnp.empty((rank, P)', 'Fv2 = fnp.empty((rv, P)'),
        ('fnp.subtract(F[:, a:a + h], F[:, a + h:a + 2 * h]',
         'fnp.subtract(F[:rv, a:a + h], F[:rv, a + h:a + 2 * h]'),
        ('return (Fu, Fv2, blocks, ru)', 'return (Fu, Fv2, blocks, ru, rank)'),
    ])
    change('_uvp_recon', 'Pv = _ubc_product', [
        ('U[:, :r_u] F_u +- U (F_v/2)', 'U[:, :r_u] F_u +- U[:, :r_v] (F_v/2)'),
        ('Fu, Fv2, blocks, ru = uv', 'Fu, Fv2, blocks, ru, rank = uv'),
        ("Pv = _ubc_product(U, Fv2, uv, 'v')", "Pv = _ubc_product(U[:, :int(Fv2.shape[0])], Fv2, uv, 'v')"),
    ])
    change('_uvp_inner', 'a_ = left_sq * lam_cols', [
        ('Fu, Fv2, blocks, ru = uv', 'Fu, Fv2, blocks, ru, rank = uv'),
        ('    del Su, Dv, inner_u\n    return inner',
         '    del Su, Dv, inner_u\n'
         '    rv = int(Fv2.shape[0])\n'
         '    if rv < rank:\n'
         '        full = fnp.zeros((rows, rank), dtype=fnp.float32)\n'
         '        _BH_HOLD.append(fnp.copyto(full[:, :rv], inner))\n'
         '        _BH_HOLD.clear()\n'
         '        inner = full\n'
         '    return inner'),
    ])
    change('_uvp_state', 'run = _UBC_RUN[0]', [
        ('Fu, Fv, blocks, ru = base', 'Fu, Fv, blocks, ru, rank = base'),
        ('generation = (tuple(blocks), int(ru), tuple(owner.shape))',
         'generation = (tuple(blocks), int(ru), int(rank), tuple(Fv.shape), tuple(owner.shape))'),
        ('tuple(Fv.shape) == (320, P)', 'tuple(Fv.shape) == (256, P)'),
    ])
    change('_ipt_forward_b_ab', "run, context = _UBC_RUN[0], _UBC_CONTEXT[0]", [
        ("elif (L == 6 and k == rank == (256 if leg == 'u' else 320)",
         "elif (leg in ('u', 'v') and L == 6 and k == rank == 256"),
    ])

    candidate = source
    for e in edits:
        candidate = replace_once(candidate, e['before'], e['after'])
    reverse = candidate
    for e in reversed(edits):
        reverse = replace_once(reverse, e['after'], e['before'])
    assert reverse == source
    changed_tree = ast.parse(candidate)
    assert len(tree.body) == len(changed_tree.body)
    changed = [(a.name, a.lineno) for a, b in zip(tree.body, changed_tree.body)
               if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False)]
    assert sorted(n for n, _ in changed) == sorted(e['function'] for e in edits)
    compile(candidate, str(D / 'candidate.py'), 'exec')

    # Pure-scalar indexing proof only: padding embeds local 256 coordinates in
    # exactly the original frame; no matrix values, imports, or models run here.
    for rank in (128, 256, 320, 384):
        ru = rv = min(256, rank)
        assert ru <= rv <= rank
        embedded = [i if i < rv else None for i in range(rank)]
        assert len(embedded) == rank and embedded[:rv] == list(range(rv))
        assert all(x is None for x in embedded[rv:])

    write('candidate.py', candidate)
    write('source_inverse.json', dict(schema='exact-source-inverse-v1', parent=pin(PARENT),
                                      candidate=pin(D / 'candidate.py'), edits=edits,
                                      inverse_passed=True, changed_functions=changed))
    write('source.diff', ''.join(difflib.unified_diff(source.splitlines(True), candidate.splitlines(True),
                                                    fromfile=str(PARENT), tofile=str(D / 'candidate.py'))))
    protocol = dict(
        schema='fixed-local-uvp-v256-proposal-v1', root_admission_pending=True,
        candidate=pin(D / 'candidate.py'), baseline=pin(PARENT), build=pin(Path(__file__).resolve()),
        source_inverse=pin(D / 'source_inverse.json'),
        fixed_scope=dict(v_rank_cap=256, common_rank_cap=256, active_layers=[10, 11, 12, 13, 14, 15],
                         retained_state_ranks=[384, 320, 384], cache_limit_bytes=512000000, cache_slots=3),
        formulas=dict(
            factors='Fu=(Ft[:ru]+Fs[:ru])/2; Fv=(Ft[:rv]-Fs[:rv])/2; ru=rv=min(256,rank).',
            reconstruction='Pt=U[:,:ru]Fu+U[:,:rv]Fv; Ps=U[:,:ru]Fu-U[:,:rv]Fv.',
            inner='Su=left_t^2*lambda_t+left_s^2*lambda_s; Dv=left_t^2*lambda_t-left_s^2*lambda_s; inner[:rv]=Dv@Fv.T; inner[:ru]+=Su@Fu.T; trailing coordinates zero.',
            lazy='Zero-embedded full-rank inner enters the unchanged full group.map.T; full mapped U, carrier Q/F/core and signed lambda remain unchanged.'),
        diagonal_and_tail=dict(
            hidden='Existing sum(pp^3*lambda), _ott_diagonal (including YNT wrapper), cp_pre fill_diagonal and authoritative64 K21 diagonal refills remain byte-for-byte source unchanged. Kept-row pp and hence its computed diagonal do change; no full-carrier diagonal restoration is claimed.',
            terminal='Existing terminal rowcut sum of reconstructed pp cubes remains; V256 changes this term. Star diagonal unchanged.',
            omitted='OTT tail trace and omitted-radial tail energy/sum still use full F/U. Kept-row omitted-radial contractions consume truncated pp. This is an intentional approximate local read, not preservation of every full-carrier moment.'),
        ownership='Five tuple metadata fields are consumed by the only three tuple-unpack sites. UBC owner identity remains full F, factors remain immutable, generation now includes local V shape. Forward cache rank check follows V256; transpose-cache branch remains unused and unchanged. All cache limits, setup shapes/deadline, finally cleanup, join schedules/ranks, young factors, learned parameters and tail owners remain unchanged.',
        pricing=dict(source_census=pin(H / 'hill_climb_s32_bill_census_v1/census.json'),
                     forecast_reconstruction_saved_flops=2102532787,
                     measured_v_minus_u_inner_proxy_flops=1803653264,
                     forecast_total_percent=1.484753317728647,
                     caveat='Predecessor rank35 geometry, fixed-route fit plus measured proxy; new routing, padding, cache copies and full-rank inner zero/copy costs are not fully priced. Native bill decides.'),
        historical_evidence=dict(path=str(H.parent / 'fable_covhunter_v1/RESULT.md'), section=37,
            caveat='LoopR3 terminal-only: plain256 +.908 raw points versus common256/full-v +.022; joint u192/v320 +.415 is not an isolated current V trial. Expected accuracy benefit is not claimed.'),
        restrictions=['Exactly one fixed V256 trial; no V-rank/window/cache variants after failure.',
                      'No model or array execution by builder; root decides native admission and gates.',
                      'No broad/serving/submission authority and no claimed improvement.'],
        source_checks=dict(inverse=True, compile_only=True, changed_functions=changed,
                           scalar_rank_embedding=True, numerical_execution=False))
    write('protocol.json', protocol)
    write('build.json', dict(passed=True, candidate=pin(D / 'candidate.py'), baseline=pin(PARENT),
                             builder=pin(Path(__file__).resolve()), protocol=pin(D / 'protocol.json'),
                             source_inverse=pin(D / 'source_inverse.json'), diff=pin(D / 'source.diff'),
                             numerical_execution=False))
    print(json.dumps(dict(candidate=pin(D / 'candidate.py'), build=pin(D / 'build.json'),
                          protocol=pin(D / 'protocol.json')), indent=2))


if __name__ == '__main__':
    main()
