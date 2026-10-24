"""TAG opus_engineer a42: the trainer's hand-over of a head column -> the builder's weights file (text only, nothing is run natively).
Two sources:
  weights_from_trainer.py <hand-over directory under O/trainer/candidates> <name>
      reads <hand-over>/u_128.npy (128 float32 weights of the new first-layer column) and column.json, and checks the weights against
      the hand-over's own enc0_w_127.npy (column 30 of the 128 x 127 array must be these weights);
  weights_from_trainer.py --json <weights json under O/trainer> <row> <name>
      reads row <row> of the trainer's (columns x 128) weights array (O/trainer/cert/head/*_weights.json: row 6 = the diagonal
      column in the native convention); every other row must be zero.
  weights_from_trainer.py --json <weights json under O/trainer> all <name>
      SEVERAL columns: every non-zero row of the array, named by its index in the trainer's HEAD_COLUMNS (6 diag, 9 k31l_diag_rows,
      12 k3_young, 13 k3_old, 14 k3_star, 15 k3_bform, 16 k4_local, 20 k4_local_rows, 21 m2, 27 qold, 28 eold, 29 qb at layers 1-14;
      17 f15_diag, 18 f15_k31l_diag, 19 f15_k4_local at layer 15; any other non-zero row has no native form yet and is refused).
      Writes {"columns": [...], "u": [[128] x K]} for the builder's lever HC@<name> (fragment v2) and, when rows 17-19 have weights,
      <name>_f15.json for the lever HCF@<name>_f15 (the final layer's columns; one weight matrix in the card, rows by layer).
      A table of 36 rows (the trainer's HSPLIT / HBANDS, 10-10 09:15Z / 09:35Z) may also carry rows 31 / 32 (columns 6 / 20 at layers 10-14:
      their weights there, row 6 / 20 then holds layers 1-9 only) and rows 33 / 34 / 35 (the constant 1 at layers 1-9 / 10-14 / 15: a shift
      of enc0_b by band); these go to a49_band/weights/<name>.json for the lever BAND@<name> (full biases = the card's enc0_b + the shift,
      in float32; the card's enc0_b is read from --candidate <candidate.py>, required then).
Writes a42_head/weights/<name>.json = {"u": [128 values], ...} for the builder's lever HC@<name>. An existing file must hold the
same weights."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
O = HERE.parents[1]
if sys.argv[1] == '--json':
    src = (O/'trainer'/sys.argv[2]) if not sys.argv[2].startswith('/') else Path(sys.argv[2])
    name = sys.argv[4]
    table = np.asarray(json.loads(src.read_text()), dtype=np.float64)
    assert table.ndim == 2 and table.shape[1] == 128, table.shape
    if sys.argv[3] == 'all':
        NATIVE = {6: 'diag', 9: 'k31l_diag_rows', 12: 'k3_young', 13: 'k3_old', 14: 'k3_star', 15: 'k3_bform', 16: 'k4_local', 20: 'k4_local_rows',
                  21: 'm2', 27: 'qold', 28: 'eold', 29: 'qb'}
        FINAL = {17: 'f15_diag', 18: 'f15_k31l_diag', 19: 'f15_k4_local'}
        BAND_ROWS = (31, 32, 33, 34, 35)
        used_all = [k for k in range(table.shape[0]) if np.any(table[k])]
        assert used_all and all(k in NATIVE or k in FINAL or k in BAND_ROWS for k in used_all), 'rows %s have weights; native columns exist for %s' % (used_all, sorted(NATIVE) + sorted(FINAL) + list(BAND_ROWS))
        final = [k for k in used_all if k in FINAL]
        if final:   # the final layer's columns: their own file for the HCF lever
            UF = table[final].astype(np.float32)
            fout = dict(columns=[FINAL[k] for k in final], u=[[float(x) for x in u] for u in UF], rows=final, source=str(src),
                        sha256_source=hashlib.sha256(src.read_bytes()).hexdigest())
            fpath = HERE/'weights'/(name + '_f15.json')
            if fpath.exists():
                old_f = json.loads(fpath.read_text())
                assert old_f.get('columns') == fout['columns'] and old_f['u'] == fout['u'], '%s exists with other weights: choose another name' % fpath
            else:
                fpath.write_text(json.dumps(fout, indent=1) + '\n')
            print(json.dumps(dict(name=name + '_f15', columns=fout['columns'], file=str(fpath))))
        used = [k for k in used_all if k in NATIVE]
        assert used, 'no column of layers 1-14 has weights'
        late = {31: 6, 32: 20}
        bands = [k for k in (31, 32, 33, 34, 35) if k < table.shape[0] and np.any(table[k])]
        if bands:   # per-band head parameters (lever BAND@<name>)
            assert '--candidate' in sys.argv, 'rows %s need the card\'s enc0_b: --candidate <candidate.py>' % bands
            import base64, re, zlib
            text = Path(sys.argv[sys.argv.index('--candidate') + 1]).read_text()
            lit = eval(re.search(r"^_GNN_LITERAL = (\{.*?\})\n", text, re.M | re.S).group(1))
            shape, blob = lit['enc0_b']
            enc0_b = np.frombuffer(zlib.decompress(base64.b64decode(blob)), dtype=np.float32).reshape(shape)
            names = [NATIVE[k] for k in used]
            for k, base_row in late.items():
                assert k not in bands or base_row in used, 'row %d (late weights of column %d) needs row %d' % (k, base_row, base_row)
            u_late = None
            if any(k in bands for k in late):
                u_late = [[float(x) for x in (table[[r for r, b in late.items() if b == k][0]] if any(b == k and r in bands for r, b in late.items()) else table[k]).astype(np.float32)] for k in used]
            enc = [None if r not in bands else [float(x) for x in (enc0_b + table[r].astype(np.float32)).astype(np.float32)] for r in (33, 34, 35)]
            bout = dict(edges=[[1, 9], [10, 14], [15, 15]], columns=names, enc0_b=enc if any(e is not None for e in enc) else None,
                        u=None if u_late is None else [None, u_late, None], rows=bands, source=str(src), sha256_source=hashlib.sha256(src.read_bytes()).hexdigest(),
                        note='rows 33-35 = enc0_b shifts by band (full biases written); rows 31 / 32 = columns 6 / 20 at layers 10-14')
            bpath = HERE.parent/'a49_band'/'weights'/(name + '.json')
            if bpath.exists():
                old_b = json.loads(bpath.read_text())
                assert old_b.get('enc0_b') == bout['enc0_b'] and old_b.get('u') == bout['u'], '%s exists with other values: choose another name' % bpath
            else:
                bpath.write_text(json.dumps(bout, indent=1) + '\n')
            print(json.dumps(dict(name=name, band_rows=bands, file=str(bpath))))
        U = table[used].astype(np.float32)
        assert float(np.abs(U.astype(np.float64) - table[used]).max()) <= 1e-7 * float(np.abs(table[used]).max())
        out = dict(columns=[NATIVE[k] for k in used], u=[[float(x) for x in u] for u in U], lengths=[float(np.linalg.norm(u.astype(np.float64))) for u in U],
                   source=str(src), rows=used, sha256_source=hashlib.sha256(src.read_bytes()).hexdigest())
        path = HERE/'weights'/(name + '.json')
        if path.exists():
            old = json.loads(path.read_text())
            assert old.get('columns') == out['columns'] and old['u'] == out['u'], '%s exists with other weights: choose another name' % path
        else:
            path.write_text(json.dumps(out, indent=1) + '\n')
        print(json.dumps(dict(name=name, columns=out['columns'], lengths=[round(x, 4) for x in out['lengths']], file=str(path))))
        sys.exit(0)
    row = int(sys.argv[3])
    assert not np.any(np.delete(table, row, axis=0)), 'another column has weights too: this fragment carries one column'
    u = table[row].astype(np.float32)
    assert float(np.abs(u.astype(np.float64) - table[row]).max()) <= 1e-7 * float(np.abs(table[row]).max()), 'the json values are not float32 values to rounding'   # the json holds more digits; the card's weights are their float32 values (s1: equal to the hand-over's u_128.npy)
    extra = dict(source=str(src), row=row, sha256_source=hashlib.sha256(src.read_bytes()).hexdigest())
else:
    src = O/'trainer'/'candidates'/sys.argv[1]
    name = sys.argv[2]
    u = np.load(src/'u_128.npy')
    assert u.shape == (128,) and u.dtype == np.float32, (u.shape, u.dtype)
    w = np.load(src/'enc0_w_127.npy')
    assert w.shape == (128, 127) and np.array_equal(w[:, 30], u), 'column 30 of enc0_w_127 is not u_128'
    extra = dict(source=str(src), sha256_u_128_npy=hashlib.sha256((src/'u_128.npy').read_bytes()).hexdigest(),
                 card=json.loads((src/'column.json').read_text()).get('card'))
values = [float(x) for x in u]
assert np.array_equal(np.asarray(values, dtype=np.float32), u)
out = dict(u=values, length=float(np.linalg.norm(u.astype(np.float64))), **extra)
path = HERE/'weights'/(name + '.json')
if path.exists():
    assert json.loads(path.read_text())['u'] == values, '%s exists with other weights: choose another name' % path
else:
    path.write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(dict(name=name, weights=len(values), length=round(out['length'], 4), largest=round(float(np.abs(u).max()), 4), file=str(path))))
