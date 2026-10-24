#!/usr/bin/env python3
"""TAG fable-engineer (E-9): stackable builder (refuses to overwrite <out.py>). Appends the engineer's fragments to ANY base candidate (the bank, or the output of
fable_engineer2_v1/build_e2.py); appended definitions override earlier ones, nothing is edited in place.
usage: build_eng.py <base.py> <out.py> [--grouped] [--read5] [--push64] [--round N]
  --grouped  grouped-request Winograd kernel (_packed_product_cached, _w5_port_product_body) with last-level copy elimination
  --v5 / --v7 / --v7b / --v7c  use earlier fragments instead of v9 (v7c = held out= aliases; v9 = v7c + views bound once per process)
  --dense    v9d instead of v9: backward corners written singly, exact columns below 32 floats, exact rows when a region would be
             at most half full (faster numpy on the D6 pushes; more view binding on the first network)
  --read5    grouped three-packet D5 read (_packed_read5); requires --grouped
  --holdelim the bank's own kernel + held aliases + last-level copy elimination (bit-exact; leaf moved to the arena tail when needed)
  --holdonly the bank's own kernel with held out= aliases only (bit-exact, same bill and compute, fewer round trips)
  --bigarena grow the arena so the six-level pushes' leaf fits the free tail (copy elimination on D6 pushes; +<=482 MB RSS)
  --sq6      allow a sixth Winograd level for square products when the depth rule says so (rounding-level change)
  --push64   zero-pad k,n of 1024-row pushes to multiples of 64 when the padded depth-6 product wins the depth rule (_mm)
  --round N  select() rounds selected counts up to a multiple of N (bank 32)
  --depthrule FRAGMENT  compute-budgeted depth rule (E-28): five-level pushes, depth-6 squares, per-shape depths (replaces --sq6/--push64)
  --f32 SETS  float64 -> float32 body patches from build_f32.py (E-29; e.g. third or third,bd), applied to the base text first
  --evict5    drop the five-level push layout's frames that the previous prediction did not use (E-31; needs --depthrule)
  --evictall  drop every frame-cache entry that the previous prediction did not use (E-31; all layouts)
  --evict5r   drop the five-level push frames and the L15 row cut's frames that the previous prediction did not use elsewhere
              (E-31 v2; needs --depthrule; the row-cut part applies when build_l15_rowcut.py ran first)
  --lru       threshold LRU over all frame caches: above 60k cached arrays at a prediction start, drop the least recently
              used entries down to 40k (E-31 v3; combines with --evict5/--evict5r, not with --evictall)
  --caps      gradual LRU entry caps on every frame cache, parents refresh children (E-32; combines with --evict5)
  --batch     batch each layer's five FLMS covariance/star 1024^3 products into two P-batched kernel calls (E-35; bit-exact)
  --bindhold  hold frame-binding intermediates (arena slices, discarded level arrays) until the kernel call ends (E-36; bit-exact)
  --caps2     with --caps: the v3 limits (about 1.1x the 20-net entry counts: a 20-net worker never evicts, the store stays bounded near 80k)
  --coldbind  cheaper frame binding (E-38; bit-exact): no reshapes for the unused arena region views, read5 child frames bound once
  --coldbind2 --coldbind + used-slot last level (E-39; bit-exact): where both sides always drop their last-level copies, bind only the
              last-level slots the kernel touches (A: 2, 4-6; B: 3-6)
  --coldbind3 --coldbind2 + scratchless unwind (E-40; bit-exact): the unwind's temporaries s0, s1, s2 are written into the products
              p1, p0, p5 (each dead by then), so the output frames no longer bind the unwind scratch (same ops, same order)
  --setupbind  Estimator.setup (off-budget, once per worker) binds the 25 frames whose shapes are fixed by the width and the kernel
              tiles, largest first within a 2 s wall; bit-exact (E-42; needs --holdelim, --depthrule, --batch)
  --symout    symmetric-output covariance (E-43 v3; rounding-level): w^T(Cw) by upper blocks through plain _mm, G21 mirrored;
              no einsum (v1/v2's einsum items withdrawn after a SymmetryError) (needs --batch)
  --lean      fewer round trips (E-45 v3; bit-exact): kernel quadrants/corners by fnp.split, function-scoped holds in select/finite
              guard/_sym_upper and the read5 packet slices; no lifetime extension (needs --holdelim)"""
import sys, hashlib, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("out")
ap.add_argument("--grouped", action="store_true"); ap.add_argument("--v5", action="store_true"); ap.add_argument("--v7", action="store_true"); ap.add_argument("--v7b", action="store_true"); ap.add_argument("--v7c", action="store_true"); ap.add_argument("--dense", action="store_true"); ap.add_argument("--bigarena", action="store_true"); ap.add_argument("--holdonly", action="store_true"); ap.add_argument("--holdelim", action="store_true"); ap.add_argument("--sq6", action="store_true"); ap.add_argument("--read5", action="store_true"); ap.add_argument("--round", type=int, default=0); ap.add_argument("--push64", action="store_true"); ap.add_argument("--depthrule", default=None); ap.add_argument("--f32", default=None); ap.add_argument("--evict5", action="store_true"); ap.add_argument("--evictall", action="store_true"); ap.add_argument("--evict5r", action="store_true"); ap.add_argument("--lru", action="store_true"); ap.add_argument("--caps", action="store_true"); ap.add_argument("--batch", action="store_true"); ap.add_argument("--bindhold", action="store_true"); ap.add_argument("--caps2", action="store_true"); ap.add_argument("--coldbind", action="store_true"); ap.add_argument("--coldbind2", action="store_true"); ap.add_argument("--coldbind3", action="store_true"); ap.add_argument("--setupbind", action="store_true"); ap.add_argument("--symout", action="store_true"); ap.add_argument("--lean", action="store_true")
args = ap.parse_args()
src = Path(args.base).read_text()
def need(name):
    assert f"def {name}(" in src, f"base lacks {name}"
parts = []
if args.grouped:
    for f in ("_packed_product_cached", "_packed_frame", "_mm", "_square_product", "_rule_depth", "_w5_port_product_body"):
        need(f)
    parts.append((HERE / ("grouped_fragment_v5.py" if args.v5 else "grouped_fragment_v7.py" if args.v7 else "grouped_fragment_v7b.py" if args.v7b else "grouped_fragment_v7c.py" if args.v7c else "grouped_fragment_v9d.py" if args.dense else "grouped_fragment_v9.py")).read_text())
if args.read5:
    assert args.grouped, "--read5 needs --grouped"
    for f in ("_packed_read5", "_read5_view"):
        need(f)
    parts.append((HERE / ("grouped_read5_fragment_v5.py" if args.v5 else "grouped_read5_fragment_v7.py" if args.v7 else "grouped_read5_fragment_v7b.py" if args.v7b else "grouped_read5_fragment_v7c.py" if args.v7c else "grouped_read5_fragment_v9.py")).read_text())
if args.holdelim:
    assert not args.grouped and not args.read5 and not args.holdonly, "--holdelim replaces the kernel"
    for f in ("_packed_product_cached", "_read5_leaf_packets", "_packed_read5", "_w5_port_product_body", "_packed_frame", "_exact_view", "_read5_packet_frames", "_a_frame", "_b_frame"):
        need(f)
    parts.append((HERE / "holdelim_fragment.py").read_text())
if args.holdonly:
    assert not args.grouped and not args.read5, "--holdonly replaces the grouped kernel"
    for f in ("_packed_product_cached", "_read5_leaf_packets", "_packed_read5", "_w5_port_product_body"):
        need(f)
    parts.append((HERE / "holdonly_fragment.py").read_text())
if args.bigarena:
    assert args.grouped or args.holdelim, "--bigarena needs --grouped or --holdelim"
    need("_predict_binary")
    parts.append((HERE / "bigarena_fragment.py").read_text())
if args.sq6:
    for f in ("_rule_depth", "_square_schedule", "_square_product", "_square_schedule_reference"):
        need(f)
    parts.append((HERE / "sq6_fragment.py").read_text())
if args.push64:
    for f in ("_mm", "_rule_depth", "_packed_product_4", "_packed_read5", "_square_product"):
        need(f)
    parts.append((HERE / "push64_fragment.py").read_text())
if args.depthrule:
    assert not args.sq6 and not args.push64, "--depthrule replaces --sq6/--push64"
    for f in ("_mm", "_rule_depth", "_square_product", "_square_schedule", "_packed_product_4", "_packed_read5", "_cp3_bind_arena_views"):
        need(f)
    assert "_PACKED_SQUARE6 =" in src and "_W5_PORT_SCOPE" in src
    dr = Path(args.depthrule)
    dr = dr if dr.exists() else (HERE / dr if (HERE / dr).exists() else HERE / dr.name)   # else resolve in fable_engineer_v1/ by name
    parts.append(dr.read_text())
if args.evict5:
    assert args.depthrule, "--evict5 needs --depthrule (the five-level push layout)"
    for f in ("_bh_fwd", "_bh_o_frame", "_bh_leafx", "_w5_port_new"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    parts.append((HERE / "evict_fragment_push5.py").read_text())
if args.evictall:
    assert not args.evict5, "--evictall includes --evict5"
    for f in ("_bh_fwd", "_bh_o_frame", "_bh_leafx", "_packed_frame", "_read5_packet_frames", "_w5_port_new"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    parts.append((HERE / "evict_fragment_all.py").read_text())
if args.evict5r:
    assert args.depthrule and not args.evict5 and not args.evictall, "--evict5r needs --depthrule and replaces --evict5/--evictall"
    for f in ("_bh_fwd", "_bh_o_frame", "_bh_leafx", "_packed_frame", "_read5_packet_frames", "_w5_port_new"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    parts.append((HERE / "evict_fragment_push5r.py").read_text())
if args.lru:
    assert not args.evictall, "--lru and --evictall are alternatives"
    for f in ("_bh_fwd", "_bh_o_frame", "_bh_leafx", "_packed_frame", "_read5_packet_frames", "_w5_port_new"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    parts.append((HERE / "evict_fragment_lru.py").read_text())
assert not args.caps2 or args.caps, "--caps2 modifies --caps"
if args.caps:
    assert not args.lru and not args.evictall, "--caps replaces --lru/--evictall"
    for f in ("_cache_put", "_bh_put", "_bh_fwd", "_packed_frame", "_read5_packet_frames"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    parts.append((HERE / ("caps_fragment_v3.py" if args.caps2 else "caps_fragment.py")).read_text())   # caps_fragment_v2.py (x1.2) superseded, never built
if args.round:
    assert args.round % 32 == 0
    need("select")
    parts.append((HERE / "select_round_fragment.py").read_text() + f"_SELECT_ROUND = {args.round}\n")
body = src
if args.f32:
    sys.path.insert(0, str(HERE)); import build_f32
    body = build_f32.apply(src, args.f32.split(","))
if args.batch:
    for f in ("_flms_covariance", "_flms_star", "_bh_fwd", "_bh_leaf", "_bh_leafx", "_square_schedule", "_exact_view"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    edits = (("            if l == 0:\n                Cpre = _resident_gram512(w32.T, w32).astype(fnp.float64)",
              "            _bb_star_pre = None   # E-35: set by the batched covariance/star call\n            if l == 0:\n                Cpre = _resident_gram512(w32.T, w32).astype(fnp.float64)"),
             ("                    Cpre = _flms_covariance(w32, C)\n                    Bpre = Cpre",
              "                    Cpre, _bb_star_pre = _flms_cov_star(w32, C, star)   # E-35: batched with the layer's star products\n                    Bpre = Cpre"),
             ("                        (_flms_star(w32, star) if _FULL_LOWER_MOMENTS else",
              "                        ((_bb_star_pre if _bb_star_pre is not None else _flms_star(w32, star)) if _FULL_LOWER_MOMENTS else"))
    for old, new in edits:
        assert body.count(old) == 1, ("batch anchor", old[:60])
        body = body.replace(old, new)
    parts.append((HERE / "batch_fragment.py").read_text())
if args.bindhold:
    for f in ("_exact_view", "_read5_view", "_cp3_bind_arena_views"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    assert "_BH_HOLD = []" in src + "".join(parts), "--bindhold needs the held-alias kernel (--holdelim)"
    parts.append((HERE / "bindhold_fragment.py").read_text())
if args.coldbind or args.coldbind2 or args.coldbind3:
    assert args.holdelim, "--coldbind replaces the held-alias read5 leaf packets (--holdelim)"
    for f in ("_cp3_bind_arena_views", "_read5_child_frame", "_read5_leaf_packets", "_read5_packet_frames", "_read5_view"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    parts.append((HERE / "coldbind_fragment.py").read_text())
if args.coldbind2 or args.coldbind3:
    for f in ("_a_frame", "_b_frame", "_exact_view", "_cache_put", "_bh_leafx", "_bh_o_frame"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    parts.append((HERE / "usedslots_fragment.py").read_text())
if args.symout:
    assert args.batch, "--symout replaces the batched _flms_cov_star (--batch)"
    for f in ("_flms_cov_star", "_mm_batch", "_mm", "_shared_old384_push", "_resident_gram512"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    parts.append((HERE / "symout_fragment_v3.py").read_text())   # v3: covariance only; v1/v2 (with the einsum items) withdrawn
if args.setupbind:
    assert args.holdelim and args.depthrule and args.batch, "--setupbind binds the holdelim, push5 and batched frames"
    for f in ("_bh_fwd", "_bh_o_frame", "_bh_leafx", "_bb_views", "_bb_o_frame", "_bb_level1", "_packed_frame", "_read5_packet_frames", "_bind_push6"):
        assert f"def {f}(" in src + "".join(parts), f"missing {f}"
    assert "class Estimator(" in src and "def setup(" not in src, "the base must not define its own setup"
    parts.append((HERE / "setupbind_fragment_v3.py").read_text())   # v1 (25 frames) + v2 (pair-map arena) superseded: v3 skips the P=3 batch under --symout
if args.lean:
    assert args.holdelim, "--lean defers the held-alias kernel's releases (--holdelim)"
    parts.append((HERE / "lean_fragment_v3.py").read_text())   # v3: no lifetime extension (v2: live peak +1.47 GB, server CPU +9-13 s); v1/v2 withdrawn
out = body + "".join(parts)
if args.coldbind3:
    # E-40: in the unwind (tl=p0+p1; s0=p0+p5; s1=s0+p6; s2=s0+p4; tr=s2+p2; bl=s1-p3; br=s1+p4) p1 is dead once tl is written, p0 and
    # p5 once s0 is, so s0, s1, s2 go to p1, p0, p5 and the output frames (_o_frame, _bh_o_frame, _bb_o_frame) skip the scratch region
    new = "        scratch_parts = (products[1], products[0], products[5])   # E-40 --coldbind3: unwind temporaries s0, s1, s2 in the dead p1, p0, p5\n"
    for old, count in (("        scratch = _exact_view(arena, u_specs[unwind], (rows, columns))\n        scratch_parts = fnp.unstack(scratch, axis=0)\n", 2),
                       ("        scratch_parts = fnp.unstack(_exact_view(arena, u_specs[unwind], (rows, columns)), axis=0)\n", 1 if args.batch else 0)):
        assert out.count(old) == count, ("coldbind3 anchor", old[:60], out.count(old))
        out = out.replace(old, new)
# E-43 v3: the v1/v2 call-site edits (layer-0 Gram einsum, old-tier joined-core einsum) are withdrawn: an f32 einsum whose output
# flopscope infers symmetric is validated with allclose(R, R.T, atol=1e-6) and raised SymmetryError on a net (S8, r100, .venv)
if args.lean:
    # E-45: kernel quadrants/corners by fnp.split; the live loop's layer boundary; the read5 packet slices held
    lean_edits = [
        ("            a11, a12, a21, a22 = a[:m2, :k2], a[:m2, k2:], a[m2:, :k2], a[m2:, k2:]\n            h((a11, a12, a21, a22))\n"
         "            b11, b12, b21, b22 = b[:k2, :n2], b[:k2, n2:], b[k2:, :n2], b[k2:, n2:]\n            h((b11, b12, b21, b22))\n",
         "            a_top, a_bot = fnp.split(a, 2, axis=0)   # E-45 --lean: quadrants by split, 3 requests per operand instead of 4\n"
         "            (a11, a12), (a21, a22) = fnp.split(a_top, 2, axis=1), fnp.split(a_bot, 2, axis=1)\n"
         "            h((a_top, a_bot, a11, a12, a21, a22))\n"
         "            b_top, b_bot = fnp.split(b, 2, axis=0)\n"
         "            (b11, b12), (b21, b22) = fnp.split(b_top, 2, axis=1), fnp.split(b_bot, 2, axis=1)\n"
         "            h((b_top, b_bot, b11, b12, b21, b22))\n", 1),
        ("            corners = (out[:rows, :columns], out[:rows, columns:], out[rows:, :columns], out[rows:, columns:])\n            h(corners)\n",
         "            o_top, o_bot = fnp.split(out, 2, axis=0)   # E-45 --lean: corners by split\n"
         "            corners = tuple(fnp.split(o_top, 2, axis=1)) + tuple(fnp.split(o_bot, 2, axis=1))\n"
         "            h((o_top, o_bot, corners))\n", 1),
        ("        a11, a12, a21, a22 = a[:m2, :k2], a[:m2, k2:], a[m2:, :k2], a[m2:, k2:]\n        b11, b12, b21, b22 = b[:k2, :n2], b[:k2, n2:], b[k2:, :n2], b[k2:, n2:]\n"
         "        h((a11, a12, a21, a22, b11, b12, b21, b22))\n",
         "        a_top, a_bot = fnp.split(a, 2, axis=0)   # E-45 --lean: quadrants by split\n"
         "        (a11, a12), (a21, a22) = fnp.split(a_top, 2, axis=1), fnp.split(a_bot, 2, axis=1)\n"
         "        b_top, b_bot = fnp.split(b, 2, axis=0)\n"
         "        (b11, b12), (b21, b22) = fnp.split(b_top, 2, axis=1), fnp.split(b_bot, 2, axis=1)\n"
         "        h((a_top, a_bot, b_top, b_bot, a11, a12, a21, a22, b11, b12, b21, b22))\n", 1 if args.batch else 0),
        ("        tl, tr, bl, br = out[:rows, :columns], out[:rows, columns:], out[rows:, :columns], out[rows:, columns:]\n        h((tl, tr, bl, br))\n",
         "        o_top, o_bot = fnp.split(out, 2, axis=0)   # E-45 --lean: corners by split\n"
         "        (tl, tr), (bl, br) = fnp.split(o_top, 2, axis=1), fnp.split(o_bot, 2, axis=1)\n"
         "        h((o_top, o_bot, tl, tr, bl, br))\n", 1 if args.batch else 0),
        ("        a, b, out = leaf_a[begin:end], leaf_b[begin:end], leaf[begin:end]\n",
         "        a, b, out = leaf_a[begin:end], leaf_b[begin:end], leaf[begin:end]\n        _BH_HOLD.append((a, b, out))   # E-45 --lean\n", 1),
    ]
    for old, new, count in lean_edits:
        assert out.count(old) == count, ("lean anchor", old[:60], out.count(old))
        out = out.replace(old, new)
    # (v2 also inserted a layer-boundary release and a loop-array snapshot in _predict_binary: withdrawn in v3)
assert not Path(args.out).exists(), f"refusing to overwrite {args.out}"
Path(args.out).write_text(out)
print(args.out, hashlib.sha256(out.encode()).hexdigest(), "base", hashlib.sha256(src.encode()).hexdigest()[:16])
