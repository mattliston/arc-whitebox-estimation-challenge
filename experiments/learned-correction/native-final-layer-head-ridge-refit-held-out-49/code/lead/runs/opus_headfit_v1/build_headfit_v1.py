"""[opus-headfit] values-only candidate: S2tr 46e9315d + a fragment that REASSIGNS _GNN_LITERAL['head'] and ['head_b'] (same shapes) with
layer 15 replaced by the frozen refit (REFIT_head15_v1.npz); every other layer and every other parameter byte-identical. _GNN_PARAMS is
unpacked lazily at first use (after import), so the patched literal is what setup and predict see."""
import ast, base64, hashlib, zlib, re
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; O = H.parents[2]
src = (O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
i = src.index('\n_GNN_LITERAL = ') + 1; j = src.index('\n', i); lit = ast.literal_eval(src[i + len('_GNN_LITERAL = '):j])
dec = lambda shape, blob: np.frombuffer(zlib.decompress(base64.b64decode(blob, validate=True)), dtype=np.float32).reshape(shape).copy()
enc = lambda a: base64.b64encode(zlib.compress(np.ascontiguousarray(a, dtype=np.float32).tobytes(), 9)).decode()
z = np.load(H/'REFIT_head15_v1.npz'); th = z['theta']
(hs, hblob), (bs, bblob) = lit['head'], lit['head_b']
head, hb = dec(hs, hblob), dec(bs, bblob)
assert head.shape[0] == 16 and head[15].size == th.size - 1, (head.shape, th.shape)
assert np.allclose(np.ravel(head[15]), z['theta0'][:-1]) and np.isclose(float(np.ravel(hb[15])[0] if np.ndim(hb[15]) else hb[15]), z['theta0'][-1])
new_head, new_hb = head.copy(), hb.copy()
new_head[15] = np.asarray(th[:-1], dtype=np.float32).reshape(head[15].shape)
new_hb[15] = np.asarray(th[-1], dtype=np.float32).reshape(np.shape(hb[15]))
assert np.array_equal(new_head[:15], head[:15]) and np.array_equal(new_hb[:15], hb[:15])
frag = (f"\n\n# BEGIN OPUS-HEADFIT FRAGMENT headfit_v1 (append LAST to S2tr 46e9315d): values only. The final layer's learned mean correction\n"
        f"# head[15] / head_b[15] replaced by a ridge least-squares refit on {len(z['train'])} fresh native networks (lambda {float(z['lam'])} by network-grouped CV);\n"
        f"# layers 0-14 and every other GNN parameter byte-identical. _GNN_PARAMS is unpacked lazily from this literal at first use.\n"
        f"_GNN_LITERAL['head'] = ({tuple(hs)!r}, {enc(new_head)!r})\n_GNN_LITERAL['head_b'] = ({tuple(bs)!r}, {enc(new_hb)!r})\n"
        f"assert _GNN_PARAMS is None\n# END OPUS-HEADFIT FRAGMENT headfit_v1\n")
out = src + frag; (H/'cand_s2tr_headfit_v1.py').write_text(out); (H/'fragment_headfit_v1.py.txt').write_text(frag)
print('candidate', hashlib.sha256(out.encode()).hexdigest()[:12], 'fragment', hashlib.sha256(frag.encode()).hexdigest()[:12], len(frag), 'bytes')
