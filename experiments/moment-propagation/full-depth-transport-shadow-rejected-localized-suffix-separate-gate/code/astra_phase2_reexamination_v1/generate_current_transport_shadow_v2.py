"""Correct reported-output parity; preserve baseline before response work."""
from pathlib import Path

here = Path(__file__).resolve().parent
source = (here/'generate_current_transport_shadow_v1.py').read_text()
source = source.replace('rows_current_transport_shadow_v1','rows_current_transport_shadow_v2')
start = source.index('    old = HERE /')
end = source.index('    if args.responses:\n        np.savez',start)
old_block = source[start:end]
save_block = '''        old = HERE / f'compact_feed_fullpair_closedloop_r{args.rank}_v1.npz'
        parity = None
        if old.exists():
            with np.load(old) as z:
                parity = float(np.max(np.abs(np.maximum(baseline,0)-z['mean'])))
                feature_parity = float(np.max(np.abs(np.stack(saved['x'])-z['x'])))
                head_parity = float(np.max(np.abs(np.stack(saved['delta'])-z['delta'])))
            print(json.dumps(dict(rank=args.rank,reported_parent_max_abs=parity,
                                  feature_max_abs=feature_parity,head_max_abs=head_parity)),flush=True)
            assert parity < 1e-12 and feature_parity == 0 and head_parity == 0
        out.mkdir(parents=True)
        for k,v in saved.items():
            np.save(out/f'{k}.npy',np.stack(v))
        np.save(out/'weights.npy',W.astype(np.float32))
        np.save(out/'baseline.npy',baseline)
'''
assert source.count('        baseline = run(capture=True)\n') == 1
source = source.replace('        baseline = run(capture=True)\n',
                        '        baseline = run(capture=True)\n'+save_block)
source = source.replace(old_block,'')
exec(compile(source,__file__,'exec'))
