"""TAG opus_engineer stack builder: RRPI + any subset of the engineer lane's levers, one text per candidate.
usage: build.py <spec> [<parent.py> <tag>]     spec = levers joined by '+', e.g.  T+C+B+F+D+Hfull   or   T+C+B+F+D+Hpad+V   or   T+C+B+F+D
  With <parent.py> <tag> the levers are applied to that file instead of RRPI (for a candidate the trainer's exporter wrote from RRPI:
  same source, other parameter values); the output is stack/candidate_<spec>__<tag>.py. Every edit still has to match exactly once.
  T      float32 tail traces (edits_lib.a08_edits)                                      rounding level
  C      one-solve Nystrom core, float32 solve (edits_lib.a08_edits, q32)               same matrix by algebra, rounding level
  B      birth update in float32 (a10_bd32/fragment_bd32.py + two loop edits)           rounding level
  F      mean-correction feature reductions with float32 accumulation (edits_lib)       rounding level
  D      live rows by slice where the dead-side cut is active (a12_live_slice)          bit-identical
  Hfull  half-zero strips, also products the parent keeps at 5 levels (hzs ks v6 + gen v7, read price 250k)   rounding level
  Hpad   half-zero strips, only products the parent already zero-pads (prices 750k)                           rounding level
  HV     with Hfull or Hpad: the strips' leaf views by one index expression each (hzs ks v7: the same views, fewer than half
         the index requests; a REQUEST lever)                                               predictions and bill unchanged
  V      bare tree views bound in setup (a11_bare_views/fragment_bvp.py)                bit-identical; setup flops up
  V<s>   the same with <s> seconds of the setup guard instead of 0.45, e.g. V0.3
  F4     the physicist's fourth-order column shift with the cheap star anchor (a14_f4/fragment_f4.py + three loop edits), strength 0.4,
         columns at most 2 layers old; F4t<theta> / F4t<theta>a<age> for other settings (F4t0 runs the step with strength 0:
         predictions equal the build without it)                                          an ERROR lever: predictions move by design
  F4v2   the same without the cohort born at layer 0 (its birth scalars are exactly zero): fragment_f4_v2.py; F4v2t<theta>a<age>
  F4v4   v3 with one strength per column age: F4v4t<theta age 1>,<theta age 2> (a single value = v3's arithmetic)
  K4Lv3  K4L v2 with one strength per pair age: K4Lv3c<c age 1>,<c age 2>
  YNT<z>  the young-column dead-tail row cut at alpha < -<z> instead of -2.5 (layers 7-14; e.g. YNT3.0): the rows between the two
         thresholds are pushed again, so their third cumulant is exact instead of the trace estimate (a22_cutsim: this band carries
         nearly all of the cut's error)                                                    an ERROR lever; bill up by the rows kept
  DS<scale>  a global scale on the learned mean correction, layers 1-15 (the lead's 05:32Z ask; torch: the 'delta' hook times <scale>):
         the per-layer gain constant of the correction net is multiplied by <scale> where it is formed (once per layer and process),
         so the correction's last multiplication carries it: no extra op per prediction      an ERROR lever; bill unchanged
  GN<knob><value>  a covariance-output knob of the parent at another value: GNiso<v>, GNk21n<v>, GNk31<v> (the parent's _GN_KNOBS literal is
         iso -0.094, k21n -0.028, k31 -0.212; one lever per knob; values as plain decimals, a leading minus allowed), for a refit
         that trains them as global dials                                                  an ERROR lever; bill unchanged
  L3     three-request operand loads of the alt-basis tree kernel (a21_load3/fragment_load3.py; no loop edit): the key's four level-1
         data slots as one view, one copy of the whole operand, the alt-basis sum taken in place. About -1,800 requests per network
         (RRPI count), bill about +0.06%                                                  bit-identical
  L3o    with L3: the upper two output quadrants of a tree product written by one pair addition (a21_load3/fragment_load3o.py):
         two requests fewer per product, bill unchanged                                   bit-identical
  K22s<scale>  the trainer's run M dial k22: the OFF-DIAGONAL part of the isotropic (2,2) plane the pair map receives times <scale>,
         layers 1-14 (one scalar multiplication per layer; the diagonal is filled separately as before)   an ERROR lever
  E2s<x>  the lead's second-order skew term of the mean formula (O/e16b_k4_closed_loop.py arm e2; the trainer's leaf e2): the mean of
         layers 1-15 gets <x> k3^2 (alpha^4 - 6 alpha^2 + 3) pdf / (72 sigma^5) through the correction slot (a29_e2/fragment_e2_v1.py +
         two loop lines); E2s<15 values, layer 1 first> gives one strength per layer            an ERROR lever; bill 20 kflop per layer
  K4Lv4  K4L v3 with the per-age multiplication always made (a strength of exactly 1 included): the operations do not depend on the
         strengths' values; a single value is written as the per-age pair
  NB<forms>[d<dfit>][x<dscale>][s<strength>][t<transport>][c<source>][b]  the bank channel of the fourth-order norm coupling folded
         into the bank read (the physicist's ncar_fold, level 'read', dresp): a second weight per bank column; <forms> 0-3 = what
         is removed; <dfit> 1 | 3 regressors of the diagonal (default 3); <dscale> factor on the multiple of tr C x C (default
         1.41); strength at layers 2-14 (default 1); b = the card's own beta as the conversion vector (no extra pass over the
         columns) (a37_fold/fragment_nb_v3.py, or fragment_nb_v4.py with the suffix v4: the same arithmetic with fewer requests; three loop
         edits, one more in the radial source with b).
  NP[s<strength>][g<diag>][a][f<forms>][r<dfit>][b][k<k31>][m][v<n>][@<dials>]  the pair law of the fourth-order norm coupling (the lead's ncar6, level 'read'): a second weight
         per bank column and a cross weight per pair, the law's sources at the birth hubs, the fold in the read (a38_pairlaw/
         fragment_np_v<n>.py; three loop edits). g0 = without the diagonal source; a = with the isotropic-rest sandwich. Not with NB.
         v2 = v1's arithmetic with a quarter of the requests (the fold takes the read's response rows itself; the law is two
         polynomial tables); v1 is the parity version the first sessions ran. v3 = v2 plus: f<forms> 1-3 = which forms the fold
         removes (1: the multiple of C; 2: + mu mu'; 3: + mu (C mu)' + sym, with the diagonal regression's coefficients); r3 = the
         regression with a constant and no table of multiples (r2, the default: the table); b = the card's beta in place of V for
         the carried weights (the radial source's column sums; one more edit, in the live radial source). v4 = v3 with the partner
         positions from two slices instead of a roll (the same values; no operation the parent does not already use but prod).
         v5 = v4's arithmetic with faster kernels for three passes over the columns (the exchange of the pair columns by split +
         concatenate, B's diagonal by a product and a matrix-vector product, the f-scaling in place): same bill, less time.
         v6 = v5 plus k<k31>: the fourth-chaos plane of the young pairs' local quartic in the same left factor (the lead's k31l.py,
         chaos4, level 'read'; cohorts at most two layers old; needs the K4L lever, whose birth coefficients it uses); m = the
         pair means over the response rows instead of all units (no pass over the full columns).
         v7 = v6 with the plane's per-column polynomial in Horner form and its coefficient table in float32 (less bill, same arithmetic).
         v8 = v7 with the trainer's dials: in the dials file K3L may be a list of strengths by cohort age (age 1 first) and DSCALE a
         list per layer (a38_pairlaw/dials_from_trainer.py turns the trainer's physics_dials.json into such a file).
         @<dials> (v3, v4): literal values from a38_pairlaw/dials/<dials>.json, {"STRENGTH": [16 values], "RHO": [16], "DSCALE": x,
         "BIRTHS": x, "ROWS": x, "DIAG": x, "CONV": x, "CA": x, "FORMS": n, "DFIT": n, "BETA": bool, ..}: any _NP_<NAME> literal of the
         fragment; they override the switches in the lever string and keep the file name short.
  LU<r>  the rank of the old pairs' common part at layers 10-14 (_LCR_RU; the parent: 256).
  JK<r7>,<r10>,<r13>  the rank of the old tier's frame at the three joins (multiples of 64; the parent: 384,320,384).
  JR[<x>]  join repair: the third-order norm coupling a join's projection drops is put back through the transported star
         of the join layer, times <x> (default 1) (a35_join/fragment_jr_v1.py; two text edits: the join and the live loop).
  SS<f>  the transported star with exact row sums (a33_starsum/fragment_ss_v3.py; no loop edit). <f> = a rank (the parent's
         factors at that rank plus the missed row sums), 0 (the row-sum part alone) or d (one direction along the row sums);
         SS<f>x<g>x<l>: form <f> for the stars moved into layers below <l>, form <g> from layer <l> on (e.g. SSdx8x12).
  SX<l>  the low-rank star's factorization with one thin product less from layer <l> on (a31_star/fragment_sx_v1.py; no loop edit):
         A ~ (A Q1) Q1' in place of Q (Q' A); bill -0.026% per layer taken                 an APPROXIMATION of the star's rank-32 form
  C2K<b>  the lead's row-energy term of the per-unit fourth cumulant (O/e16b_k4_closed_loop.py arm c2k=<b>, E33): the pair map of layers
         1-14 gets k4_i += (6 b / n) (C^2)_ii with the isotropic level projected out (a32_c2k/fragment_c2k_v1.py + one loop line)
                                                                                          an ERROR lever; bill 2 n^2 flops per layer
  BR<x>  the lead's birth dial (O/e27_birth_dials.sh, arm br=<x>): the birth coefficient of the newborn pairs d (1 + <x> radial / 3)
         instead of d (1 + radial / 3), every layer; F4's birth scalar keeps its own (1 + radial / 3), as in the lead's torch arm
         (one loop line; no op added)                                                     an ERROR lever; bill unchanged
  F4v3   v2's arithmetic with every temporary held, plus the physicist's two optional dials: F4v3t<theta>a<age>o<offset>k<kappa>,
         offset e = birth scalar A + e / sigma (default 0), kappa = scale of the b carrier's source term radial d v / 6 (default 1)
  F4C    the physicist's exact-term mixture (O/physicist/f4c_native_ref.py; a27_f4c/fragment_f4c_v1.py + F4's three loop edits + one
         birth call): F4's rule at strength theta plus the hub's exact S1, S2 at c1, c2, six coefficients per pair and age fixed at
         the pair's birth. F4Ct<theta>c<c1>,<c2> (defaults 0.378 and 0.48,0.48); t<age 1>,<age 2> and c<c1 a1>,<c2 a1>,<c1 a2>,<c2 a2>
         give one set per age. Replaces F4 (not both).                                     an ERROR lever: predictions move by design
  F4B    DRAFT: the physicist's F4b (two shifts per pair from the hub's exact fourth cumulant; a18_f4b/fragment_f4b.py + four loop edits),
         exact mode, strength 0.5, pairs at most 2 layers old, star anchor; F4Bc<strength>a<age>t<terms>l<lo>-<hi>, t<terms> = which
         of S1..S4 are kept (t12 = S1 and S2), l<lo>-<hi> = layer range of the shift, a trailing b = anchor on the carried b vector.
         Replaces F4 (not both).                                                           an ERROR lever: predictions move by design
  K4L    the lead's non-isotropic per-unit fourth cumulant from the young pairs' hub coefficients (a15_k4l/fragment_k4l.py + five
         edits), strength 1, pairs at most 2 layers old; K4Lc<c> / K4Lc<c>a<age> for other settings;
         K4Lv2 = the same arithmetic with every temporary held and lower-case einsum subscripts (fragment_k4l_v2.py)
                                                                                          an ERROR lever: predictions move by design
  E      the learned covariance edge's tanh (0.715% of the bill at 16 flops per element) by a stand-in exact to 2e-4 on pointwise ops
         (a17_tanh/fragment_ets.py, 10 flops per element), and the edge's bias added before the (n, n, 16) broadcast instead of after
         (one tensor pass fewer); Etanh / Ebias for one of the two alone                    an approximation (2e-4) / rounding level
  Ecut   the learned covariance edge evaluated on the response rows x alive columns only (a19_edge_cut/fragment_ecut.py + one loop
         line that publishes the two rosters + the edge function's tail); EcutS = the same with the tanh stand-in. Not with E.
                                                                                          an approximation of one learned part
  Z<cut><z>  AUDIT ONLY: one of the parent's row-cut thresholds set to <z> (99 = that cut off): Zott (old tier, positive tail, parent 2.5),
           Zneg (old tier, dead negative tail, 2.5), Zynt (young columns, dead negative tail, 2.5), Zdead (dead-side cut of the push, 3.5),
           Zl15_ (final-layer k3 row cut on |mu/sigma|, 3.5); e.g. Zott99+Zneg99, Zdead4.0. Changes predictions and bill.
  P<name>  a regenerated prebind list, a13_prebind/prebind_<name>.py (bit-identical; which tree keys setup binds and fuses)
  SO2m<c>[r<r>]  the second-order mean term inside the pair map at strength c, E relu^2 term at r (default c): a41_base/fragment_so2_v1.py
         (physicist's so2.py, mean / raw2; no loop edit)                                   a physics change (refit base)
  SO2m<c>[r<r>][t<t>[d][k<0|1>][p<digits>]][f<f>]  with t or f: a41_base/fragment_so2_v2.py = v1 + the second-order part of the third-cumulant target at
         strength t (d = its diagonal alone; k = highest power of rho kept, p = the parts a b q c d e as digits 1-6; default k1 p123456) + the mean term at the
         final layer at strength f (one loop line). The physicist's pbase keywords so2m so2t so2td so2k so2p so2f.
  HC@<name>  the trainer's new input column of the learned mean correction's first layer (the unit-specific part of the pair law's diagonal on the
         read's response rows, standardised; 128 weights from a42_head/weights/<name>.json): a42_head/fragment_hc_v1.py + two lines of the fold (pair law v9)
         + one line of gnn_step. No bill worth naming, 19-21 requests a layer.
  BSK<x> the birth weight with the skew-corrected kink density at strength x: a41_base/fragment_bsk_v1.py + one loop line
         (physicist's arm bsk)                                                             a physics change (refit base)
  GSK<s>[k<k>][b<b>]  the transport gate with the first-order corrections of the mean gate: skew at strength s, kurtosis at k, the
         b-mode feed at b (a41_base/fragment_gsk_v1.py + five loop edits; physicist's arms gsk / gk4 / gbk)   a physics change (refit base)
  BK3p<x1>,<x2>[k<k>] / BK3r<x1>,<x2>[k<k>]  the collective mode times the specific third cumulant in the (3,1) plane (physicist's bk3.py) at
         level 'pair' (a row factor on the bank + star planes and the K21 plane's outer product: one loop edit) or 'r31' (the pair
         map's extra plane, through the loop's hook); a41_base/fragment_bk3_v1.py                    a physics change (refit base)
  BT@<name>  the base-term levers listed in a41_base/bases/<name>.json ({"levers": [...]}); the file name keeps BT@<name>
  DCV<r>[f<floor>]  the covariance sandwich of layers 1-15 with the units of the layer below under the gate floor cut (the roster rounded up to a
         multiple of r, default 64; floor default the live roster's 0.01): a44_cov/fragment_dcv_v1.py (v2 with f) + two call-site edits;
         DCV<r>f<floor><g|p|t>: v3, the cut product's kernel generic / padded specialised / tiles at the inner size
                                                                                          an approximation (2e-3 of C_pre at 0.01)
  L15@<name>  layer 15's own copy of the learned correction's encoder / messages / readout (a45_l15/fragment_l15_v1.py + four lines
         of gnn_step; arrays from a45_l15/weights/<name>.npz, any subset; L15@off = the switch with no weights)   OFF: bit-identical
  KC@<name>  the trainer's kappa_C per layer (a47_kc/kc/<name>.json = {"kc": {layer: value}}): the post covariance's rank-2 update after the
         learned mean correction scaled by (1 + rc_l - kc_l) instead of (1 + rc_l) (a47_kc/fragment_kc_v1.py + the loop's two lines)   all 0: bit-identical
  BAND@<name>  per-band enc0 bias and head-column weights (a49_band/weights/<name>.json = {"edges": [[1, 9], [10, 14], [15, 15]],
         "enc0_b": [128-vector or null per band] or null, "u": [K x 128 or null per band] or null, "columns": the HC lever's};
         a49_band/fragment_band_v1.py, a wrapper of _hc_bias, outermost)                       OFF: bit-identical; on: free
  TRK<r7>,<r10>,<r13>  the old tier's rank at the joins of layers 7 / 10 / 13 (the literal _R320_RULE; the card's 384,320,384)   a physics change
  OVR@<name>  literal overrides appended LAST (after the release fragments): a50_ovr/<name>.json = {"_NAME": value, ...}; for knobs read at
         call time (the kernels' request and time prices); asserts every name exists in the file                 values only
  XRQ      request diet of the pair series (a56_req): byte-identical, about -500 requests per network
  HZ7 | HZ7g<x>  7-level strip pushes (a57_hz7): taken where the bill model gains more than 3% (HZ7) or x (HZ7g0.0 = any gain)
  XRQS     persistent scratch (a56_req/fragment_rqs_v1.py): kept tile / group outputs of the symmetric sandwich and their views; byte-identical
  HZD      direct strip loads of the k-split reads (a59_direct/fragment_hzd_v1.py): no zero-interleaved pack; byte-identical, bill down
  HZD2     = HZD + the depth-6 strip products (a59_direct/fragment_hzd_v2.py, one loader for every strip layout)
  HZD3     = HZD2 with the quadrant-view cache bounded (a59_direct/fragment_hzd_v3.py)
  XSDM     the dead cut's symmetric product at its inner size m (a60_psdm/fragment_psdm_v1.py; needs XRQS and DCV ..p); rounding-level
  U15a | U15<p>.<p>  the final layer's masked inputs 13-15 / 26 from the full C_pre15 (a55_u15; needs DCV): all four, or those listed
  DLY@<name>  per-layer values of the trainer's global dials (a52_pld/values/<name>.json = {"bscale" | "X1" | "iso" | "k4la1": {layer: value}}):
         a52_pld/fragment_pld_v1.py + the use sites (bscale in the loop, X1 in the birth update, iso in _gn_dcov)   values only
  CS     the once-per-process constants and buffers built at the head of the binding phase instead of inside the first network
         (a40_setup/fragment_cs_v1.py; no loop edit; appended after the prebind list, before the release fragments)   bit-identical
  R<name>  a release-lane fragment from O/release/variants/fragments/<name>.py, appended last (e.g. Rlvfuse_sized32, Rtsr_off)
Output: stack/candidate_<spec with '+' -> '_'>.py and its .build.json (parent sha, every edit, appended text sha)."""
import sys
from pathlib import Path
S = Path(__file__).resolve().parent
E = S.parent
sys.path.insert(0, str(E)); sys.path.insert(0, str(E/'a11_bare_views'))
from buildlib import build, RRPI, RRPI_SHA
from edits_lib import a08_edits, features_f32_edits
from analyze_binds import FAMILIES
spec = sys.argv[1]
PARENT, PARENT_SHA, TAG = (Path(sys.argv[2]).resolve(), None, '__' + sys.argv[3]) if len(sys.argv) > 2 else (RRPI, RRPI_SHA, '')
levers = spec.split('+')
for _bt in [x for x in levers if x.startswith('BT@')]:   # a set of base-term levers from a file (short file names): a41_base/bases/<name>.json = {"levers": ["SO2m1", "BSK1", ...]}
    _terms = __import__('json').loads((E/('a41_base/bases/%s.json' % _bt[3:])).read_text())['levers']
    levers = levers[:levers.index(_bt)] + list(_terms) + levers[levers.index(_bt) + 1:]
known = {'T', 'C', 'B', 'F', 'D', 'Hfull', 'Hpad', 'HV', 'HV2', 'E', 'Etanh', 'Ebias', 'Ecut', 'EcutS', 'Ecutv2', 'EcutSv2', 'L3', 'L3o', 'CS'}
assert all(x in known or x[0] in 'VPRZ' or x.startswith('F4') or x.startswith('K4L') or x.startswith('SO2m') or x.startswith('HC@') or x.startswith('HCF@') or x.startswith('SRO@') or x.startswith('BSK') or x.startswith('GSK') or x.startswith('BK3') or x.startswith('K22s') or x.startswith('YNT') or x.startswith('GN') or x.startswith('DS') or x.startswith('BR') or x.startswith('SS') or x.startswith('JR') or x.startswith('JK') or x.startswith('LU') or x.startswith('NB') or x.startswith('NP') or x.startswith('E2s') or x.startswith('C2K') or x.startswith('SX') or x.startswith('DCV') or x.startswith('L15@') or x.startswith('KC@') or x.startswith('BAND@') or x.startswith('TRK') or x.startswith('OVR@') or x.startswith('DLY@') or x.startswith('U15') or x.startswith('HZ7') or x in ('XRQ', 'XRQ2', 'XRQS', 'HZD', 'HZD2', 'HZD3', 'XSDM') for x in levers) and len(set(levers)) == len(levers), levers
assert not ('Hfull' in levers and 'Hpad' in levers)
assert not ('HV' in levers or 'HV2' in levers) or 'Hfull' in levers or 'Hpad' in levers
assert not ('HV' in levers and 'HV2' in levers)   # HV2 = HV with the leaf views cached per tree (fragment_ks_v8.py)
edits = a08_edits('1' if 'T' in levers else '0', 'q32' if 'C' in levers else 'none')
parts = []
if 'F' in levers:
    edits += features_f32_edits(PARENT.read_text())
if 'B' in levers:
    edits += [
        ("                bp, Cpre, K21, aux, d, alive, births, a, t, s, _bd_current_c)\n",
         "                _ps_bp32, Cpre, K21, aux, d, alive, births, a, t, s, _bd_current_c)   # opus-engineer bd32: the float32 copy made three lines up\n", 1),
        ("            born = fnp.add(born, _bd_plane_delta, dtype=fnp.float64, out=born)\n",
         "            born = fnp.add(born, _bd_plane_delta, out=born)   # opus-engineer bd32: both float32 (was widened to float64 and stored in float32)\n", 1),
    ]
    parts.append((E/'a10_bd32/fragment_bd32.py').read_text().strip('\n'))
if 'D' in levers:
    edits.append(("                right = _unique_rows_via_scatter(pp[:, int(shared_old['columns']):], live_indices)   # E2-R: only the young columns of right are ever read\n",
                  "                right = _dcr_right(pp, int(shared_old['columns']), live_indices)   # E2-R: only the young columns of right are ever read (opus-engineer dcr: a slice of the pushed rows when the rosters agree)\n", 1))
    parts.append((E/'a12_live_slice/fragment_dcr.py').read_text().strip('\n'))
hzs = 'full' if 'Hfull' in levers else ('pad' if 'Hpad' in levers else None)
if hzs:
    hz = E/'hzs'
    d5 = 'True' if hzs == 'full' else 'False'
    ks = (hz/('fragment_ks_v8.py' if 'HV2' in levers else ('fragment_ks_v7.py' if 'HV' in levers else 'fragment_ks_v6.py'))).read_text()
    assert ks.count('_HZ_PRICE = 750000.0') == 1 and ks.count('_HZ_D5 = True ') == 1
    parts.append(ks.replace('_HZ_PRICE = 750000.0', '_HZ_PRICE = %s.0' % ('250000' if hzs == 'full' else '750000'))
                 .replace('_HZ_D5 = True ', '_HZ_D5 = %s ' % d5).strip('\n'))
    gen = (hz/'fragment_gen_v7.py').read_text()
    assert gen.count('_HZG_PRICE = 750000.0') == 1 and gen.count('_HZG_D5 = True ') == 1
    parts.append(gen.replace('_HZG_D5 = True ', '_HZG_D5 = %s ' % d5).strip('\n'))
ecut = [x for x in levers if x in ('Ecut', 'EcutS', 'Ecutv2', 'EcutSv2')]
if ecut:
    assert len(ecut) == 1 and not [x for x in levers if x in ('E', 'Etanh', 'Ebias')]
    edits += [
        ("            _R320_RESP[l] = int(response_indices.shape[0])   # PT-60: resp_L, before this layer's push/join\n",
         "            _R320_RESP[l] = int(response_indices.shape[0])   # PT-60: resp_L, before this layer's push/join\n"
         "            _E2_LAYER.update(ecut_rows=response_indices, ecut_cols=live_indices)   # opus-engineer Ecut: the rosters the covariance edge keeps\n", 1),
        ("    h = _e2_H(_e2_H(left[:, None, :]) + right) + t['bias']\n"
         "    _e2_hold.append(fnp.add(h,k['R3']*k['e0'],out=h))\n    _e2_hold.append(fnp.add(h,k['S3']*k['e1'],out=h))\n"
         "    _e2_hold.append(fnp.add(h,k['ST3']*k['e2'],out=h))\n    sf=s.astype(fnp.float32);pf=p.astype(fnp.float32);gf=g.astype(fnp.float32)\n"
         "    _e2_hold.append(fnp.tanh(h,out=h))\n"
         "    raw = fnp.einsum('ij,i,j->ij', _e2_H(h @ t['output']), pf, gf, optimize=False)\n"
         "    return _e2_H(_e2_H(_e2_H(0.5 * _e2_H(raw + _e2_H(raw.T))) * _e2_H(fnp.outer(sf, sf))) * t['cov_factor'])\n",
         "    sf=s.astype(fnp.float32);pf=p.astype(fnp.float32);gf=g.astype(fnp.float32)\n"
         "    raw = _ecut_raw(left, right, t, k, pf, gf, _e2_hold)   # opus-engineer Ecut: the edge on the response rows x alive columns, 0 elsewhere\n"
         "    return _e2_H(_e2_H(_e2_H(0.5 * _e2_H(raw + _e2_H(raw.T))) * _e2_H(fnp.outer(sf, sf))) * t['cov_factor'])\n", 1),   # the live definition only
    ]
    frag = (E/('a19_edge_cut/fragment_ecut_v2.py' if ecut[0].endswith('v2') else 'a19_edge_cut/fragment_ecut.py')).read_text()   # v2: the same values bit for bit, 7 requests and 9 Mflop fewer a layer
    assert frag.count('_ECUT_STANDIN = False\n') == 1
    parts.append(frag.replace('_ECUT_STANDIN = False\n', '_ECUT_STANDIN = %r\n' % ecut[0].startswith('EcutS')).strip('\n'))
edge = [x for x in levers if x in ('E', 'Etanh', 'Ebias')]
if edge:
    assert len(edge) == 1
    head = "    h = _e2_H(_e2_H(left[:, None, :]) + right) + t['bias']\n"
    mid = ("    _e2_hold.append(fnp.add(h,k['R3']*k['e0'],out=h))\n    _e2_hold.append(fnp.add(h,k['S3']*k['e1'],out=h))\n"
           "    _e2_hold.append(fnp.add(h,k['ST3']*k['e2'],out=h))\n    sf=s.astype(fnp.float32);pf=p.astype(fnp.float32);gf=g.astype(fnp.float32)\n")
    tanh = "    _e2_hold.append(fnp.tanh(h,out=h))\n"
    tail = ("    raw = fnp.einsum('ij,i,j->ij', _e2_H(h @ t['output']), pf, gf, optimize=False)\n"
            "    return _e2_H(_e2_H(_e2_H(0.5 * _e2_H(raw + _e2_H(raw.T))) * _e2_H(fnp.outer(sf, sf))) * t['cov_factor'])\n")
    new_head = ("    h = _e2_H(_e2_H(left + t['bias'])[:, None, :]) + right   # opus-engineer E: the bias joins the (n, 16) operand, one (n, n, 16) pass fewer\n"
                if edge[0] in ('E', 'Ebias') else head)
    new_tanh = ("    _e2_hold.append(_ets_tanh(h, _e2_hold))   # opus-engineer E: tanh by its 10-flop stand-in (2e-4)\n" if edge[0] in ('E', 'Etanh') else tanh)
    edits.append((head + mid + tanh + tail, new_head + mid + new_tanh + tail, 1))   # the live definition only (the older one ends in .astype)
    if edge[0] in ('E', 'Etanh'):
        parts.append((E/'a17_tanh/fragment_ets.py').read_text().strip('\n'))
f4b = [x for x in levers if x.startswith('F4B')]
if f4b:
    import re
    assert len(f4b) == 1 and not [x for x in levers if x.startswith('F4') and not x.startswith('F4B')]
    match = re.fullmatch(r'F4B(?:v([0-9]+))?(?:c([0-9.]+))?(?:a([0-9]+))?(?:t([1-4]+))?(?:l([0-9]+)-([0-9]+))?(b)?', f4b[0])
    assert match, f4b
    bversion = int(match.group(1) or 1)          # v2: the same arithmetic on the whole block of shifted columns, one shift block, half the charged ops
    strength, age = float(match.group(2) or 0.5), int(match.group(3) or 2)
    terms = tuple(int(ch) for ch in (match.group(4) or '1234'))
    lo, hi = int(match.group(5) or 1), int(match.group(6) or 14)
    anchor = 'b' if match.group(7) else 'star'
    shift_lines = ("                for _f4b_off, _f4b_blk in _f4b[0]:\n"
                   "                    _f4b_c0 = int(p.shape[1]) - _f4b_off\n"
                   "                    _f4b_py = _e2_H(p[:, _f4b_c0:_f4b_c0 + int(_f4b_blk.shape[1])])\n"
                   "                    _e2_H(fnp.add(_f4b_py, _f4b_blk, out=_f4b_py))\n" if bversion == 1 else
                   "                _f4b_py = _e2_H(p[:, int(p.shape[1]) - int(_f4b[0].shape[1]):])\n"
                   "                _e2_H(fnp.add(_f4b_py, _f4b[0], out=_f4b_py))\n")
    edits += [
        ("            trace_post = fnp.outer(_ps32(_ps_Bdiag), _ps32(b)) + fnp.einsum('ij,i->ij', _ps32(Bpost), _ps32(2 * b), optimize=False)\n",
         "            _f4b = _f4b_step(l, right, g, d, live_indices, _ps_Bdiag, lam, shared_old) if len(lam) else None   # opus-engineer F4b: two shifts per young pair and their anchor\n"
         + ("            _f4_b = b if _f4b is None else b + _f4b[1]   # star anchor: the star's trace term takes b + beta; the carried b and the births keep b\n" if anchor == 'star' else
            "            _f4_b = b   # b anchor: the star's trace term keeps b; the carried b is lowered at the end of the step\n")
         + "            trace_post = fnp.outer(_ps32(_ps_Bdiag), _ps32(_f4_b)) + fnp.einsum('ij,i->ij', _ps32(Bpost), _ps32(2 * _f4_b), optimize=False)\n", 1),
        ("            p = fnp.einsum('i,ij->ij', _e2_ga[live_indices], right if len(lam) else pp[live_indices, int(shared_old['columns']):], optimize=False)   # E-48 --colpack E4: the gated young columns on the rows the next push reads\n",
         "            p = fnp.einsum('i,ij->ij', _e2_ga[live_indices], right if len(lam) else pp[live_indices, int(shared_old['columns']):], optimize=False)   # E-48 --colpack E4: the gated young columns on the rows the next push reads\n"
         "            if _f4b is not None:   # opus-engineer F4b: shift the young pairs' gated columns in place\n"
         + shift_lines, 1),
        ("            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag\n",
         "            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag\n"
         "            if _f4b is not None:   # opus-engineer F4b: the exact diagonal of the plane the shifts add\n"
         + ("                _ps_rest_diag = _ps_rest_diag - _f4b[2]\n" if anchor == 'star' else
            "                _ps_rest_diag = _ps_rest_diag - _e2_H(_f4b[2] - _e2_H(_e2_H(3 * _ps_Bdiag) * _f4b[1]))\n"), 1),
        ("            newborn = (int(_e2_H(pp.shape[1])), selected, _e2_H(_bd_base32[:, selected]), _e2_tsel, _e2_ssel)\n",
         "            newborn = (int(_e2_H(pp.shape[1])), selected, _e2_H(_bd_base32[:, selected]), _e2_tsel, _e2_ssel)\n"
         "            _f4b_birth(l, aux, host, selected, t, s)   # opus-engineer F4b: the newborn pairs' quartic on their own two columns\n", 1),
    ]
    if anchor == 'b':
        edits.append(("            fnp.fill_diagonal(star, _ps_star_diag)\n            if collect_states:\n",
                      "            fnp.fill_diagonal(star, _ps_star_diag)\n"
                      "            if _f4b is not None:   # opus-engineer F4b, b anchor: the carried b gives up the part the shifts now hold\n"
                      "                b = b - _f4b[1]\n"
                      "            if collect_states:\n", 1))
    frag = (E/'a18_f4b'/('fragment_f4b.py' if bversion == 1 else 'fragment_f4b_v%d.py' % bversion)).read_text()
    for old, new in (("_F4B_C = 0.5\n", "_F4B_C = %r\n" % strength), ("_F4B_AGE = 2\n", "_F4B_AGE = %d\n" % age),
                     ("_F4B_ANCHOR = 'star'\n", "_F4B_ANCHOR = %r\n" % anchor), ("_F4B_LAYERS = (1, 14)\n", "_F4B_LAYERS = (%d, %d)\n" % (lo, hi)),
                     ("_F4B_TERMS = (1, 2, 3, 4)\n", "_F4B_TERMS = %r\n" % (terms,))):
        assert frag.count(old) == 1, old
        frag = frag.replace(old, new)
    parts.append(frag.strip('\n'))
f4c = [x for x in levers if x.startswith('F4C')]
if f4c:
    import re
    assert len(f4c) == 1 and not [x for x in levers if x.startswith('F4') and not x.startswith('F4C')]
    match = re.fullmatch(r'F4C(?:v([0-9]+))?(?:t([-0-9.,]+))?(?:c([-0-9.,]+))?', f4c[0])
    assert match, f4c
    cversion = int(match.group(1) or 1)
    theta = tuple(float(x) for x in (match.group(2) or '0.378').split(','))
    cs = tuple(float(x) for x in (match.group(3) or '0.48,0.48').split(','))
    assert len(theta) in (1, 2) and len(cs) in (2, 4), 'F4C: t<theta> or t<age 1>,<age 2>; c<c1>,<c2> or c<c1 age 1>,<c2 age 1>,<c1 age 2>,<c2 age 2>'
    edits += [
        ("            trace_post = fnp.outer(_ps32(_ps_Bdiag), _ps32(b)) + fnp.einsum('ij,i->ij', _ps32(Bpost), _ps32(2 * b), optimize=False)\n",
         "            _f4 = _f4c_step(l, right if len(lam) else None, g, d, live_indices, _ps_Bdiag, lam, shared_old)   # opus-engineer F4c: the shift of the young pairs' columns (rule + exact terms) and its star anchor\n"
         "            _f4_b = b if _f4 is None else b + _f4[1]   # the star's trace term takes b + beta; the carried b and the births keep b\n"
         "            trace_post = fnp.outer(_ps32(_ps_Bdiag), _ps32(_f4_b)) + fnp.einsum('ij,i->ij', _ps32(Bpost), _ps32(2 * _f4_b), optimize=False)\n", 1),
        ("            p = fnp.einsum('i,ij->ij', _e2_ga[live_indices], right if len(lam) else pp[live_indices, int(shared_old['columns']):], optimize=False)   # E-48 --colpack E4: the gated young columns on the rows the next push reads\n",
         "            p = fnp.einsum('i,ij->ij', _e2_ga[live_indices], right if len(lam) else pp[live_indices, int(shared_old['columns']):], optimize=False)   # E-48 --colpack E4: the gated young columns on the rows the next push reads\n"
         "            if _f4 is not None:   # opus-engineer F4c: shift the young gated columns in place\n"
         "                _f4_py = p[:, int(p.shape[1]) - int(_f4[0].shape[1]):]\n"
         "                _e2_H(fnp.add(_f4_py, _f4[0], out=_f4_py))\n", 1),
        ("            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag\n",
         "            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag\n"
         "            if _f4 is not None:   # opus-engineer F4c: the exact diagonal of the plane the shift adds\n"
         "                _ps_rest_diag = _ps_rest_diag - _f4[2]\n", 1),
        ("            newborn = (int(_e2_H(pp.shape[1])), selected, _e2_H(_bd_base32[:, selected]), _e2_tsel, _e2_ssel)\n",
         "            newborn = (int(_e2_H(pp.shape[1])), selected, _e2_H(_bd_base32[:, selected]), _e2_tsel, _e2_ssel)\n"
         "            _f4c_birth(l, aux, d, host, selected, t, s, newborn[2], _e2_asel, radial)   # opus-engineer F4c: the newborn pairs' six shift coefficients per age\n", 1),
    ]
    frag = (E/'a27_f4c'/('fragment_f4c_v%d.py' % cversion)).read_text()
    assert frag.count('_F4C_THETA = 0.378 ') == 1 and frag.count('_F4C_C = (0.48, 0.48) ') == 1
    # always the per-age form, also for equal strengths: two files that differ in strengths then differ in float literals only
    # (the release lane's fast path for literal-only candidates; the fragment's operations do not depend on the values)
    parts.append(frag.replace('_F4C_THETA = 0.378 ', '_F4C_THETA = %r ' % ((theta * 2)[:2],))
                 .replace('_F4C_C = (0.48, 0.48) ', '_F4C_C = %r ' % (((cs * 2)[:2], (cs * 2)[2:4]),)).strip('\n'))
f4 = [x for x in levers if x.startswith('F4') and not x.startswith(('F4B', 'F4C'))]
if f4:
    import re
    assert len(f4) == 1
    match = re.fullmatch(r'F4(?:v([0-9]+))?(?:t([0-9.,]+))?(?:a([0-9]+))?(?:o(-?[0-9.]+))?(?:k([0-9.]+))?', f4[0])
    assert match, f4
    version, age = int(match.group(1) or 1), int(match.group(3) or 2)
    theta = tuple(float(x) for x in match.group(2).split(',')) if ',' in (match.group(2) or '') else float(match.group(2) or 0.4)
    assert not isinstance(theta, tuple) or (version >= 4 and len(theta) == age), 'one strength per age needs F4v4 and as many values as ages (age 1 first)'
    offset, kappa = float(match.group(4) or 0.0), float(match.group(5) or 1.0)
    assert version >= 3 or (offset == 0.0 and kappa == 1.0)
    add = ('                fnp.add(_f4_py, _f4[0], out=_f4_py)\n' if version < 3 else
           '                _e2_H(fnp.add(_f4_py, _f4[0], out=_f4_py))\n')
    edits += [
        ("            trace_post = fnp.outer(_ps32(_ps_Bdiag), _ps32(b)) + fnp.einsum('ij,i->ij', _ps32(Bpost), _ps32(2 * b), optimize=False)\n",
         "            _f4 = _f4_step(l, right if len(lam) else None, g, d, live_indices, _ps_Bdiag, lam, shared_old, aux, selected, radial)   # opus-engineer F4: the shift of the youngest columns and its star anchor\n"
         "            _f4_b = b if _f4 is None else b + _f4[1]   # the star's trace term takes b + beta; the carried b and the births keep b\n"
         "            trace_post = fnp.outer(_ps32(_ps_Bdiag), _ps32(_f4_b)) + fnp.einsum('ij,i->ij', _ps32(Bpost), _ps32(2 * _f4_b), optimize=False)\n", 1),
        ("            p = fnp.einsum('i,ij->ij', _e2_ga[live_indices], right if len(lam) else pp[live_indices, int(shared_old['columns']):], optimize=False)   # E-48 --colpack E4: the gated young columns on the rows the next push reads\n",
         "            p = fnp.einsum('i,ij->ij', _e2_ga[live_indices], right if len(lam) else pp[live_indices, int(shared_old['columns']):], optimize=False)   # E-48 --colpack E4: the gated young columns on the rows the next push reads\n"
         "            if _f4 is not None:   # opus-engineer F4: shift the youngest gated columns in place\n"
         "                _f4_py = p[:, int(p.shape[1]) - int(_f4[0].shape[1]):]\n"
         + add, 1),
        ("            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag\n",
         "            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag\n"
         "            if _f4 is not None:   # opus-engineer F4: the exact diagonal of the plane the shift adds\n"
         "                _ps_rest_diag = _ps_rest_diag - _f4[2]\n", 1),
    ]
    if kappa != 1.0:
        edits.append(("            b = _e2_H(g * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / 6) if _FULL_LOWER_MOMENTS else ",
                      "            b = _e2_H(g * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / _F4_SOURCE6) if _FULL_LOWER_MOMENTS else ", 1))
    frag = (E/'a14_f4'/('fragment_f4.py' if version == 1 else 'fragment_f4_v%d.py' % version)).read_text()
    if version >= 3:
        assert frag.count('_F4_OFFSET = 0.0\n') == 1
        frag = frag.replace('_F4_OFFSET = 0.0\n', '_F4_OFFSET = %r\n_F4_SOURCE6 = %r   # 6 / kappa: the b carrier source radial d v / 6 times kappa (used only when the builder edits the loop line)\n' % (offset, 6.0 / kappa))
    assert frag.count('_F4_THETA = 0.4\n') == 1 and frag.count('_F4_AGE = 2\n') == 1
    parts.append(frag.replace('_F4_THETA = 0.4\n', '_F4_THETA = %r\n' % (theta,)).replace('_F4_AGE = 2\n', '_F4_AGE = %d\n' % age).strip('\n'))
k4l = [x for x in levers if x.startswith('K4L')]
if k4l:
    import re
    assert len(k4l) == 1
    match = re.fullmatch(r'K4L(?:v([0-9]+))?(?:c([0-9.,]+))?(?:a([0-9]+))?', k4l[0])
    assert match, k4l
    kversion, age = int(match.group(1) or 1), int(match.group(3) or 2)
    strength = tuple(float(x) for x in match.group(2).split(',')) if ',' in (match.group(2) or '') else float(match.group(2) or 1.0)
    assert not isinstance(strength, tuple) or (kversion >= 3 and len(strength) == age), 'one strength per age needs K4Lv3 and as many values as ages (age 1 first)'
    if kversion >= 4 and not isinstance(strength, tuple):
        strength = (strength,) * age   # v4: always the per-age form (its operations do not depend on the values)
    hold = (lambda text: text) if kversion < 2 else (lambda text: '_e2_H(%s)' % text)
    edits += [
        ("                _e2_t3 = _e2_pp2 * pp\n",
         "                _e2_t3 = _e2_pp2 * pp\n"
         "                _k4l_loc = _k4l_local(l, pp, _e2_pp2, _e2_t3, shared_old)   # opus-engineer K4L: the young pairs' hub terms of the per-unit fourth cumulant\n", 1),
        ("                k4 = k4 * (_S19_K4BAND[0] if l <= 7 else (_S19_K4BAND[1] if l <= 11 else _S19_K4BAND[2]))\n",
         "                k4 = k4 * (_S19_K4BAND[0] if l <= 7 else (_S19_K4BAND[1] if l <= 11 else _S19_K4BAND[2]))\n"
         "            if l >= 1 and len(lam):   # opus-engineer K4L: the non-isotropic part of the per-unit fourth cumulant, at the level the card was trained with\n"
         "                k4 = %s + %s\n" % (hold('k4'), hold('_k4l_dk(_k4l_loc, _e2_a, vv)')), 1),
        ("            newborn = (int(_e2_H(pp.shape[1])), selected, _e2_H(_bd_base32[:, selected]), _e2_tsel, _e2_ssel)\n",
         "            newborn = (int(_e2_H(pp.shape[1])), selected, _e2_H(_bd_base32[:, selected]), _e2_tsel, _e2_ssel)\n"
         "            _k4l_birth(l, aux, host, selected, t, s)   # opus-engineer K4L: the newborn pairs' four hub coefficients\n", 1),
        ("                m, aux = nonlinear_scalar(mu, v, k3, _e2_H(_e2_H(radial * v) * v))\n",
         "                m, aux = nonlinear_scalar(mu, v, k3, _e2_H(_e2_H(radial * v) * v) + %s)   # opus-engineer K4L: the final layer's per-unit fourth cumulant\n" % hold('_k4l_final(mu, v)'), 1),
        ("    k3_rows = fnp.sum(_cp_diagonal_cube(pp_rows) * lam, axis=1).astype(fnp.float64)   # F32 (E-29)\n    del pp_rows\n",
         "    k3_rows = fnp.sum(_cp_diagonal_cube(pp_rows) * lam, axis=1).astype(fnp.float64)   # F32 (E-29)\n"
         "    _K4L_FINAL[0] = (rows, _k4l_local(layer, pp_rows, None, None, state))   # opus-engineer K4L: the hub terms on the rows this push keeps\n"
         "    del pp_rows\n", 1),
    ]
    frag = (E/'a15_k4l'/('fragment_k4l.py' if kversion == 1 else 'fragment_k4l_v%d.py' % kversion)).read_text()
    assert frag.count('_K4L_C = 1.0\n') == 1 and frag.count('_K4L_AGE = 2\n') == 1
    parts.append(frag.replace('_K4L_C = 1.0\n', '_K4L_C = %r\n' % (strength,)).replace('_K4L_AGE = 2\n', '_K4L_AGE = %d\n' % age).strip('\n'))
k22 = [x for x in levers if x.startswith('K22s')]
if k22:
    assert len(k22) == 1
    edits.append(("            k22 = _ps32(radial) * (_ps_vv32 + 2 * _ps32(_e2_CC)) / 3\n",
                  "            k22 = _ps32(_e2_H(radial * _K22_SCALE) if 1 <= l <= 14 else radial) * (_ps_vv32 + 2 * _ps32(_e2_CC)) / 3   # opus-engineer K22: the off-diagonal part times the trained scale (the diagonal is filled on the next line)\n", 1))
    parts.append('# BEGIN OPUS-ENGINEER K22 (run M dial k22: scale of the off-diagonal part of the isotropic (2,2) plane, layers 1-14)\n'
                 '_K22_SCALE = %r\n# END OPUS-ENGINEER K22' % float(k22[0][4:]))
e2 = [x for x in levers if x.startswith('E2s')]
if e2:
    assert len(e2) == 1
    values = tuple(float(x) for x in e2[0][3:].split(','))
    assert len(values) in (1, 15), 'E2s<strength> for layers 1-15, or 15 values (layer 1 first)'
    edits += [
        ("                if str(dm.dtype) != 'float64':\n                    dm = fnp.asarray(dm, fnp.float64)\n",
         "                if str(dm.dtype) != 'float64':\n                    dm = fnp.asarray(dm, fnp.float64)\n"
         "                if l >= 1:   # opus-engineer E2: the second-order skew term of the mean, through the correction slot\n"
         "                    dm = dm + _e2_H(_e2_skew(l, aux))\n", 1),
        ("                m = m + _e2_H(final_hook(l, m, aux, _e2_H(W[l])))\n",
         "                m = m + _e2_H(_e2_H(final_hook(l, m, aux, _e2_H(W[l]))) + _e2_H(_e2_skew(l, aux)))   # opus-engineer E2: the final layer's second-order skew term\n", 1),
    ]
    frag = (E/'a29_e2/fragment_e2_v1.py').read_text()
    old = '_E2_STRENGTH = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) '
    assert frag.count(old) == 1
    parts.append(frag.replace(old, '_E2_STRENGTH = %r ' % ((0.0,) + (values * 15)[:15],)).strip('\n'))   # always 16 literals: strengths differ in float literals only
sx = [x for x in levers if x.startswith('SX')]
if sx:
    assert len(sx) == 1
    frag = (E/'a31_star/fragment_sx_v1.py').read_text()
    assert frag.count('_SX_FIRST = 4\n') == 1
    parts.append(frag.replace('_SX_FIRST = 4\n', '_SX_FIRST = %d\n' % int(sx[0][2:])).strip('\n'))
ss = [x for x in levers if x.startswith('SS')]
if ss:
    assert len(ss) == 1
    form = lambda x: -1 if x == 'd' else int(x)   # a rank, 0 = the row-sum part alone, d = the one-direction form
    spec_ss = (ss[0][2:] or '32').split('x')           # SS<form> or SS<early form>x<late form>x<first late layer>
    early, late, first_late = form(spec_ss[0]), form(spec_ss[1] if len(spec_ss) > 1 else spec_ss[0]), int(spec_ss[2]) if len(spec_ss) > 1 else 16
    assert len(spec_ss) in (1, 3) and 1 <= first_late <= 16
    frag = (E/'a33_starsum/fragment_ss_v3.py').read_text()
    old_rank = '_SS_RANK = (32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32)\n'
    assert frag.count(old_rank) == 1
    parts.append(frag.replace(old_rank, '_SS_RANK = %r\n' % (tuple(early if layer < first_late else late for layer in range(1, 16)),)).strip('\n'))   # always 15 literals: forms differ in literals only
jr = [x for x in levers if x.startswith('JR')]
if jr:
    assert len(jr) == 1
    edits += [
        ("        state['rank_new'] = rank   # PT-60: the new frame's rank\n        Fj = _shared_old384_mm(_e2_H(Qn.T), J)\n        joined_core = None\n",
         "        state['rank_new'] = rank   # PT-60: the new frame's rank\n        Fj = _shared_old384_mm(_e2_H(Qn.T), J)\n        _jr_record(layer, previous, J, coefficient, Qn, Fj)   # opus-engineer JR: the norm coupling this join drops\n        joined_core = None\n", 1),
        ("            pp = _deadcut_push(l, w32, p, previous, newborn, lam, shared_old, _e2_a)\n",
         "            pp = _deadcut_push(l, w32, p, previous, newborn, lam, shared_old, _e2_a)\n            if l in _JR_STATE:   # opus-engineer JR: what this layer's join dropped of the norm coupling goes back in through the transported star\n                _jr_apply(l, w32, _bb_star_pre, _plane_star32)\n", 1),
    ]
    frag = (E/'a35_join/fragment_jr_v1.py').read_text()
    assert frag.count('_JR_SCALE = 1.0 ') == 1
    parts.append(frag.replace('_JR_SCALE = 1.0 ', '_JR_SCALE = %r ' % float(jr[0][2:] or 1.0)).strip('\n'))
jk = [x for x in levers if x.startswith('JK')]
if jk:
    assert len(jk) == 1
    ranks = [int(x) for x in jk[0][2:].split(',')]
    assert len(ranks) == 3 and all(r % 64 == 0 and 128 <= r <= 384 for r in ranks), ranks
    parts.append('# BEGIN OPUS-ENGINEER JOIN RANKS (an ERROR-AND-BILL lever: the rank of the old tier\'s frame at the joins of layers 7, 10, 13; the parent has 384, 320, 384)\n'
                 '_R320_RULE = {7: %d, 10: %d, 13: %d}\n# END OPUS-ENGINEER JOIN RANKS' % tuple(ranks))
lu = [x for x in levers if x.startswith('LU')]
if lu:
    assert len(lu) == 1 and int(lu[0][2:]) % 64 == 0 and 128 <= int(lu[0][2:]) <= 384
    parts.append('# BEGIN OPUS-ENGINEER COMMON-PART RANK (an ERROR-AND-BILL lever: at layers 10-14 the old pairs\' common part u is rebuilt from the\n'
                 '# leading _LCR_RU directions of the frame; the parent has 256, the frame has 320 or 384)\n_LCR_RU = %d\n# END OPUS-ENGINEER COMMON-PART RANK' % int(lu[0][2:]))
nb = [x for x in levers if x.startswith('NB')]
if nb:
    assert len(nb) == 1
    m_nb = re.fullmatch(r'NB([0-3])(?:d([13]))?(?:x([0-9.]+))?(?:s([0-9.]+))?(?:t([0-9.]+))?(?:c([0-9.]+))?(b)?(?:v([34]))?', nb[0])   # forms, dfit, dscale, strength, transport, source (conversion), beta, fragment version
    assert m_nb, nb
    edits += [
        ("                response_plane = _e2_H(_cp_read_young_sq(left2, right, lam, live_indices, shared_old))\n",
         "                _nb_context = _nb_fold(l, left2, pp, response_indices, lam, mu, _e2_vc, vv, Cpre)   # opus-engineer NB: the bank channel of the norm coupling, folded into the read's left factor\n"
         "                response_plane = _e2_H(_cp_read_young_sq(left2, right, lam, live_indices, shared_old))\n", 1),
        ("                fnp.put(cp_pre, _e2_H(_e2_H(_e2_H(response_indices * n)[:, None]) + live_indices), response_plane)\n",
         "                fnp.put(cp_pre, _e2_H(_e2_H(_e2_H(response_indices * n)[:, None]) + live_indices), response_plane)\n"
         "                _nb_forms(l, cp_pre, _nb_context, response_mask, alive, mu, _plane_cov32)   # opus-engineer NB: the forms the fold must not contain, back on the read's block\n", 1),
        ("            if _S19_LAM is None:\n                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n",
         "            _nb_update(l, pp, lam, m, g, d, 2 * int(_e2_asel.shape[0]), 1.0 if _S19_LAM is None else _S19_LAM[0])   # opus-engineer NB: the second weight per bank column\n"
         "            if _S19_LAM is None:\n                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n", 1),
    ]
    if m_nb.group(7):
        # the LIVE radial source (the last definition: the one that repairs the old columns' omitted tail rows), after its repair of pp' beta
        edits.append(("        _ott_h1 = fnp.add(column_sum[:columns], delta_sum, out=column_sum[:columns])\n    contraction = fnp.sum(_e2_H(_g5_chain(_e2_H(lam64.astype(fnp.float64)), row_energy, column_sum)))\n",
                      "        _ott_h1 = fnp.add(column_sum[:columns], delta_sum, out=column_sum[:columns])\n    L['nb_column_sum'] = column_sum   # opus-engineer NB: pp' beta (tail rows of the old columns included), reused by the second weight's update\n    contraction = fnp.sum(_e2_H(_g5_chain(_e2_H(lam64.astype(fnp.float64)), row_energy, column_sum)))\n", 1))
    frag = (E/('a37_fold/fragment_nb_v%s.py' % (m_nb.group(8) or '3'))).read_text()   # v3 and v4: the same arithmetic up to rounding, v4 with fewer requests
    strength = float(m_nb.group(4) or 1.0)
    subs = [('_NB_STRENGTH = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0) ', '_NB_STRENGTH = %r ' % ((0.0, 0.0) + (strength,) * 13 + (0.0,),)),
            ('_NB_FORMS = 1\n', '_NB_FORMS = %d\n' % int(m_nb.group(1))), ('_NB_DFIT = 3\n', '_NB_DFIT = %d\n' % int(m_nb.group(2) or 3)),
            ('_NB_DSCALE = 1.41\n', '_NB_DSCALE = %r\n' % float(m_nb.group(3) or 1.41)), ('_NB_TRANSPORT = 1.0\n', '_NB_TRANSPORT = %r\n' % float(m_nb.group(5) or 1.0)),
            ('_NB_SOURCE = 1.0\n', '_NB_SOURCE = %r\n' % float(m_nb.group(6) or 1.0)), ('_NB_BETA = False\n', '_NB_BETA = %r\n' % bool(m_nb.group(7)))]
    for old_s, new_s in subs:
        assert frag.count(old_s) == 1, old_s
        frag = frag.replace(old_s, new_s)
    parts.append(frag.strip('\n'))
np_ = [x for x in levers if x.startswith('NP')]
if np_:
    assert len(np_) == 1 and not nb
    m_np = re.fullmatch(r'NP(?:s([0-9.]+))?(?:g([0-9.]+))?(a)?(?:f([123]))?(?:r([23]))?(b)?(?:k([0-9.]+))?(m)?(?:v([1-9]))?(?:@([A-Za-z0-9_]+))?', np_[0])   # strength of the fold, strength of the diagonal source, isotropic-rest sandwich, forms, regression, beta, fragment version
    assert m_np, np_
    np_version = m_np.group(9) or '1'
    np_dials = __import__('json').loads((E/('a38_pairlaw/dials/%s.json' % m_np.group(10))).read_text()) if m_np.group(10) else {}
    assert np_version in ('3', '4', '5', '6', '7', '8', '9') or not np_dials
    np_beta = bool(np_dials['BETA']) if 'BETA' in np_dials else bool(m_np.group(6))
    assert np_version in ('3', '4', '5', '6', '7', '8', '9') or not (m_np.group(4) or m_np.group(5) or m_np.group(6)), 'forms, regression and beta are switches of fragment v3'
    assert np_version in ('6', '7', '8', '9') or not (m_np.group(7) or m_np.group(8)), 'the young pairs\' plane is a switch of fragments v6, v7'
    assert not m_np.group(7) or k4l, 'the young pairs\' plane uses the K4L lever\'s birth coefficients'
    if np_version in ('3', '4', '5', '6', '7', '8', '9'):
        edits += [
            ("                left2 = _unique_rows_via_scatter(_e2_pp2, response_indices)\n",
             "                left2 = _np_left(l, _e2_pp2, pp, response_indices, response_mask, alive, lam, mu, _e2_vc, vv, Cpre)   # opus-engineer NP: the response rows of the squared columns, with the pair law of the norm coupling folded in\n", 1),
            ("                fnp.put(cp_pre, _e2_H(_e2_H(_e2_H(response_indices * n)[:, None]) + live_indices), response_plane)\n",
             "                fnp.put(cp_pre, _e2_H(_e2_H(_e2_H(response_indices * n)[:, None]) + live_indices), response_plane)\n"
             "                _np_forms(l, cp_pre, _plane_cov32)   # opus-engineer NP: the forms the fold must not contain, back on the read's block\n", 1),
            ("            if _S19_LAM is None:\n                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n",
             "            _np_update(l, lam, pp, mu, vv, m, d, aux, t, s, delta, safe_birth, _e2_vc, response_mask, selected, radial, _ps_K2132, _ps_kdiag)   # opus-engineer NP: the second weight per bank column and the cross weight per pair\n"
             "            if _S19_LAM is None:\n                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n", 1),
        ]
        if np_beta:
            edits.append(("        _ott_h1 = fnp.add(column_sum[:columns], delta_sum, out=column_sum[:columns])\n    contraction = fnp.sum(_e2_H(_g5_chain(",
                          "        _ott_h1 = fnp.add(column_sum[:columns], delta_sum, out=column_sum[:columns])\n    L['np_column_sum'] = column_sum   # opus-engineer NP: pp' beta (tail rows of the old columns included), reused by the carried weights\n    contraction = fnp.sum(_e2_H(_g5_chain(", 1))
    elif np_version == '1':
        edits += [
            ("                response_plane = _e2_H(_cp_read_young_sq(left2, right, lam, live_indices, shared_old))\n",
             "                _np_context = _np_fold(l, left2, pp, response_indices, lam, mu, _e2_vc, vv, Cpre)   # opus-engineer NP: the pair law of the norm coupling, folded into the read's left factor\n"
             "                response_plane = _e2_H(_cp_read_young_sq(left2, right, lam, live_indices, shared_old))\n", 1),
            ("                fnp.put(cp_pre, _e2_H(_e2_H(_e2_H(response_indices * n)[:, None]) + live_indices), response_plane)\n",
             "                fnp.put(cp_pre, _e2_H(_e2_H(_e2_H(response_indices * n)[:, None]) + live_indices), response_plane)\n"
             "                _np_forms(l, cp_pre, _np_context, response_mask, alive, _plane_cov32)   # opus-engineer NP: the multiple of C the fold must not contain, back on the read's block\n", 1),
            ("            if _S19_LAM is None:\n                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n",
             "            _np_update(l, _e2_H(locals()))   # opus-engineer NP: the second weight per bank column and the cross weight per pair\n"
             "            if _S19_LAM is None:\n                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n", 1),
        ]
    else:
        edits += [
            ("                left2 = _unique_rows_via_scatter(_e2_pp2, response_indices)\n",
             "                left2 = _np_left(l, _e2_pp2, pp, response_indices, response_mask, lam, mu, _e2_vc, vv, Cpre)   # opus-engineer NP: the response rows of the squared columns, with the pair law of the norm coupling folded in\n", 1),
            ("                fnp.put(cp_pre, _e2_H(_e2_H(_e2_H(response_indices * n)[:, None]) + live_indices), response_plane)\n",
             "                fnp.put(cp_pre, _e2_H(_e2_H(_e2_H(response_indices * n)[:, None]) + live_indices), response_plane)\n"
             "                _np_forms(l, cp_pre, alive, _plane_cov32)   # opus-engineer NP: the multiple of C the fold must not contain, back on the read's block\n", 1),
            ("            if _S19_LAM is None:\n                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n",
             "            _np_update(l, lam, pp, mu, vv, m, d, aux, t, s, delta, safe_birth, _e2_vc, response_mask, selected, radial, _ps_K2132, _ps_kdiag)   # opus-engineer NP: the second weight per bank column and the cross weight per pair\n"
             "            if _S19_LAM is None:\n                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n", 1),
        ]
    frag = (E/('a38_pairlaw/fragment_np_v%s.py' % np_version)).read_text()
    strength = float(m_np.group(1) or 1.0)
    subs = [('_NP_STRENGTH = (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0) ', '_NP_STRENGTH = %r ' % ((0.0,) + (strength,) * 14 + (0.0,),)),
            ('_NP_DIAG = 1.0\n', '_NP_DIAG = %r\n' % float(m_np.group(2) if m_np.group(2) is not None else 1.0)), ('_NP_AREST = False\n', '_NP_AREST = %r\n' % bool(m_np.group(3)))]
    if np_version in ('3', '4', '5', '6', '7', '8', '9'):
        subs += [('_NP_FORMS = 1\n', '_NP_FORMS = %d\n' % int(m_np.group(4) or 1)), ('_NP_DFIT = 2\n', '_NP_DFIT = %d\n' % int(m_np.group(5) or 2)),
                 ('_NP_BETA = False\n', '_NP_BETA = %r\n' % bool(m_np.group(6)))]
    if np_version in ('6', '7', '8', '9'):
        subs += [('_NP_K3L = 0.0\n', '_NP_K3L = %r\n' % float(m_np.group(7) or 0.0)), ('_NP_K3L_ALL = True\n', '_NP_K3L_ALL = %r\n' % (not m_np.group(8)))]
    for old_s, new_s in subs:
        assert frag.count(old_s) == 1, old_s
        frag = frag.replace(old_s, new_s)
    for key, value in np_dials.items():   # dial values from a file: any literal of the fragment, by name; the line's comment stays
        value = tuple(value) if isinstance(value, list) else value
        frag, count = re.subn(r'(?m)^_NP_%s = [^#\n]*?( *#.*)?$' % re.escape(key), lambda mo, key=key, value=value: '_NP_%s = %r%s' % (key, value, mo.group(1) or ''), frag)
        assert count == 1, key
    hcf_l = [x for x in levers if x.startswith('HCF@')]
    if hcf_l and set(__import__('json').loads((E/('a42_head/weights/%s.json' % hcf_l[0][4:])).read_text())['columns']) & {'f15_diag', 'f15_k31l_diag'}:
        for old_s, new_s in (("    layers = range(last)                                                # the layers with an update: 0 .. last - 1\n",
                              "    layers = range(last + 1)   # opus-engineer hcf: the update also at the last layer of the fold, for the final layer's head columns\n"),
                             ("    if l >= C['last'] or not (K or S['QX'] is not None):",
                              "    if l >= C['last'] + 1 or not (K or S['QX'] is not None):")):
            assert frag.count(old_s) == 1, old_s
            frag = frag.replace(old_s, new_s)
    hc = [x for x in levers if x.startswith('HC@')]
    if hc:   # the trainer's head columns: made inside the fold / the loop from arrays they hold, used by the correction's first layer (one line of gnn_step)
        assert len(hc) == 1 and np_version == '9', 'the head columns are written for the pair law v9'
        hc_spec = __import__('json').loads((E/('a42_head/weights/%s.json' % hc[0][3:])).read_text())
        hc_columns = hc_spec.get('columns', ['diag'])
        hc_call = ("    solution = hold(solution / hold(fnp.unstack(solution))[C['target']])                      # kap x (multiple x c0, .., c1, c2), 1, 0, ..\n",
                   "    solution = hold(solution / hold(fnp.unstack(solution))[C['target']])                      # kap x (multiple x c0, .., c1, c2), 1, 0, ..\n"
                   "    _hc_column(l, Y, C['target'], vc, response_mask, int(response_indices.shape[0]))   # opus-engineer hc: the head's new input column from this layer's diagonal\n")
        edits.append(("    h = silu(_e2_H(_e2_H(k['H'] @ k['enc0_wT']) + P['enc0_b']))\n",
                      "    h = silu(_e2_H(_e2_H(k['H'] @ k['enc0_wT']) + _e2_H(_hc_bias(l, P['enc0_b']))))   # opus-engineer hc: enc0_b + z u' when the layer's read made the new column\n", 1))
        if hc_columns == ['diag'] and not hc_spec.get('v2'):   # one column: v1, byte for byte the fired S42 arithmetic
            hc_weights = hc_spec['u']
            assert len(hc_weights) == 128
            fold_subs = [("    S['diag'] = S['pp2'] = S['fm'] = S['aa'] = S['ab'] = S['Ub'] = S['Rb'] = None\n",
                          "    S['diag'] = S['pp2'] = S['fm'] = S['aa'] = S['ab'] = S['Ub'] = S['Rb'] = None\n"
                          "    _HC_STATE['l'] = _HC_STATE['z'] = None   # opus-engineer hc: no column from an earlier read\n"), hc_call]
            hc_frag = (E/'a42_head/fragment_hc_v1.py').read_text()
            assert hc_frag.count('_HC_U = None ') == 1
            hc_frag = hc_frag.replace('_HC_U = None ', '_HC_U = %r ' % (tuple(float(x) for x in hc_weights),))
        else:   # several columns (v2): each leaves its per-unit vector; one standardisation and one product on the bias
            known_columns = ('diag', 'k31l_diag_rows', 'k3_young', 'k3_old', 'k3_star', 'k3_bform', 'k4_local', 'k4_local_rows', 'm2', 'qb', 'eold', 'qold',
                             'kmu', 'kg', 'qk', 'e34', 's1', 'fold')
            assert not ('k4_local' in hc_columns and 'k4_local_rows' in hc_columns)
            assert hc_columns and all(c in known_columns for c in hc_columns) and len(set(hc_columns)) == len(hc_columns), hc_columns
            hc_weights = hc_spec['u']
            assert len(hc_weights) == len(hc_columns) and all(len(u) == 128 for u in hc_weights)
            fold_subs = [hc_call] if 'diag' in hc_columns else []
            if 'k31l_diag_rows' in hc_columns:
                fold_subs.append(("        local = (y0, y1, value)                                                               # (k31 / 3) x (the plane's left factor) / lam, per column\n",
                                  "        local = (y0, y1, value)                                                               # (k31 / 3) x (the plane's left factor) / lam, per column\n"
                                  "        _hc_pdiag(l, value, own, hold(lam[y0:y1]), response_indices, response_mask, vc, n)   # opus-engineer hc: the plane's diagonal on the response rows\n"))
            if 'k3_bform' in hc_columns:
                edits.append(("            _e2_a = mu / _e2_s\n",
                              "            _e2_a = mu / _e2_s\n            _hc_put(l, 'k3_bform', bpre / _e2_s)   # opus-engineer hc: the b-form's per-unit third cumulant over v sigma (3 b / sigma)\n", 1))
            if 'k3_star' in hc_columns:
                edits.append(("                         _star_compact(w32, star, previous, previous_alive, live_indices)))\n            _w5_port_end_layer(_w5_context)\n",
                              "                         _star_compact(w32, star, previous, previous_alive, live_indices)))\n"
                              "            _hc_put(l, 'k3_star', fnp.diag(star_pre) / (_e2_vc * _e2_s))   # opus-engineer hc: the star's per-unit third cumulant over v sigma\n"
                              "            _w5_port_end_layer(_w5_context)\n", 1))
            if 'k3_young' in hc_columns or 'k3_old' in hc_columns:
                edits.append(("                diagonal = _e2_H(fnp.sum(_e2_t3, axis=1)).astype(fnp.float64)\n                diagonal = _ott_diagonal(diagonal, lam)\n",
                              "                _hc_before = _e2_H(fnp.sum(_e2_t3, axis=1)).astype(fnp.float64)\n                diagonal = _ott_diagonal(_hc_before, lam)\n"
                              "                _hc_k3(l, _e2_t3, int(shared_old['columns']), _hc_before, diagonal, _e2_vc, _e2_s)   # opus-engineer hc: the bank's per-unit third cumulant, young columns and old tier apart\n", 1))
            if 'k4_local' in hc_columns or 'k4_local_rows' in hc_columns:
                edits.append(("                k4 = _e2_H(k4) + _e2_H(_k4l_dk(_k4l_loc, _e2_a, vv))\n",
                              "                _hc_k4l = _e2_H(_k4l_dk(_k4l_loc, _e2_a, vv))\n                k4 = _e2_H(k4) + _hc_k4l\n"
                              + ("                _hc_put(l, 'k4_local', _hc_k4l / (_e2_vc * _e2_vc))   # opus-engineer hc: K4L's per-unit value over v^2\n" if 'k4_local' in hc_columns else
                                 "                _hc_put(l, 'k4_local_rows', _hc_k4l / (_e2_vc * _e2_vc), response_mask, int(response_indices.shape[0]))   # opus-engineer hc: K4L's per-unit value over v^2, on the response rows\n"), 1))
            hc_frag = (E/'a42_head/fragment_hc_v2.py').read_text()
            assert hc_frag.count("_HC_COLUMNS = ('diag',) ") == 1 and hc_frag.count('_HC_U = None ') == 1
            hc_frag = hc_frag.replace("_HC_COLUMNS = ('diag',) ", '_HC_COLUMNS = %r ' % (tuple(hc_columns),)).replace('_HC_U = None ', '_HC_U = %r ' % (tuple(tuple(float(x) for x in u) for u in hc_weights),))
            b7 = set(hc_columns) & {'kmu', 'kg', 'qk', 'e34', 's1', 'fold'}
            assert 'fold' not in b7, 'fold is out (lead 10-10: +1.34% of bill); fragment b7 v1 still has it'
            if set(hc_columns) & {'m2', 'qb', 'eold', 'qold'}:   # the physicist's batch-6 columns (a46_b6/fragment_b6_v1.py; v2 when qb shares c and den with qk)
                assert k4l or not set(hc_columns) & {'eold', 'qold'}, 'the old cohorts\' columns read K4L\'s birth coefficients and its call site'
                if 'm2' in hc_columns and 'e34' not in hc_columns:   # with e34, m2 comes from the shared polynomial table (fragment b7 v2)
                    edits.append(("            pair_map = _nonlinear_first_gaussian if l == 0 else _nonlinear_terminal_live if l == _e2_H(depth - 1) else nonlinear\n",
                                  "            _b6_m2(l, _e2_a, _e2_vc, _e2_s, _ps_kdiag, response_mask, int(response_indices.shape[0]))   # opus-engineer b6: the second-order mean term's column\n"
                                  "            pair_map = _nonlinear_first_gaussian if l == 0 else _nonlinear_terminal_live if l == _e2_H(depth - 1) else nonlinear\n", 1))
                if 'qb' in hc_columns:
                    edits.append(("            _E2_LAYER.update(ps_K21=K21, ps_K2132=_ps_K2132)\n",
                                  "            _E2_LAYER.update(ps_K21=K21, ps_K2132=_ps_K2132)\n"
                                  "            _b6_qb(l, _ps_cp32, _plane_cpdiag64, mu, response_gate, _e2_s, _e2_vc, response_mask, int(response_indices.shape[0]))   # opus-engineer b6: the read's plane against the scale mode's column\n", 1))
                if 'eold' in hc_columns or 'qold' in hc_columns:
                    k4l_call = "                _k4l_loc = _k4l_local(l, pp, _e2_pp2, _e2_t3, shared_old)   # opus-engineer K4L: the young pairs' hub terms of the per-unit fourth cumulant\n"
                    edits.append((k4l_call, k4l_call + "                _b6_old(l, pp, _e2_pp2, lam, shared_old, _e2_vc, response_mask, int(response_indices.shape[0]))   # opus-engineer b6: the old cohorts' energy and local quartic\n", 1))
                b6_file = 'a46_b6/fragment_b6_v2.py' if b7 & {'kmu', 'kg', 'qk'} else 'a46_b6/fragment_b6_v1.py'
                hc_frag = hc_frag.rstrip('\n') + '\n\n\n' + (E/b6_file).read_text().strip('\n')
            if b7:   # the physicist's batch-7 columns (a50_b7/fragment_b7_v2.py): one loop line each kind
                if 'e34' in b7:   # m2 and e34 from one polynomial table (before the pair map)
                    edits.append(("            pair_map = _nonlinear_first_gaussian if l == 0 else _nonlinear_terminal_live if l == _e2_H(depth - 1) else nonlinear\n",
                                  "            _b67_poly(l, _e2_a, _e2_vc, _e2_s, _ps_kdiag, k4, response_mask, int(response_indices.shape[0]))   # opus-engineer b7: m2 and e34 from one polynomial table\n"
                                  "            pair_map = _nonlinear_first_gaussian if l == 0 else _nonlinear_terminal_live if l == _e2_H(depth - 1) else nonlinear\n", 1))
                if b7 & {'kmu', 'kg', 'qk'}:   # after qb's line when qb is named (qk takes qb's c and den)
                    k21_anchor = ("            _b6_qb(l, _ps_cp32, _plane_cpdiag64, mu, response_gate, _e2_s, _e2_vc, response_mask, int(response_indices.shape[0]))   # opus-engineer b6: the read's plane against the scale mode's column\n"
                                  if 'qb' in hc_columns else "            _E2_LAYER.update(ps_K21=K21, ps_K2132=_ps_K2132)\n")
                    edits.append((k21_anchor, k21_anchor + "            _b7_k21(l, K21, _ps_kdiag, mu, response_gate, _e2_s, _e2_vc, _ps_K2132)   # opus-engineer b7: the K21 plane's row sums against mu, g, c\n", 1))
                if 's1' in b7:
                    k4l_call7 = "                _k4l_loc = _k4l_local(l, pp, _e2_pp2, _e2_t3, shared_old)   # opus-engineer K4L: the young pairs' hub terms of the per-unit fourth cumulant\n"
                    edits.append((k4l_call7, k4l_call7 + "                _b7_bank(l, pp, _e2_pp2, lam, shared_old, _e2_vc)   # opus-engineer b7: the age-1 cohort's signed energy\n", 1))
                hc_frag = hc_frag.rstrip('\n') + '\n\n\n' + (E/'a50_b7/fragment_b7_v2.py').read_text().strip('\n')
        for old_s, new_s in fold_subs:
            assert frag.count(old_s) == 1, old_s
            frag = frag.replace(old_s, new_s)
        parts.append(hc_frag.strip('\n'))
    parts.append(frag.strip('\n'))
c2k = [x for x in levers if x.startswith('C2K')]
if c2k:
    assert len(c2k) == 1
    edits.append(("            _e2_CC = Cpre * Cpre\n            _E2_LAYER.update(VV_for=Cpre, CC=_e2_CC)\n",
                  "            _e2_CC = Cpre * Cpre\n            _E2_LAYER.update(VV_for=Cpre, CC=_e2_CC)\n"
                  "            if 1 <= l <= 14:   # opus-engineer C2K: the row-energy term of the per-unit fourth cumulant\n"
                  "                k4 = _e2_H(k4) + _e2_H(_c2k_dq(_e2_CC, _e2_a, vv))\n", 1))
    frag = (E/'a32_c2k/fragment_c2k_v1.py').read_text()
    assert frag.count('_C2K_B = 0.25\n') == 1
    parts.append(frag.replace('_C2K_B = 0.25\n', '_C2K_B = %r\n' % float(c2k[0][3:])).strip('\n'))
br = [x for x in levers if x.startswith('BR')]
if br:
    assert len(br) == 1
    edits.append(("            birth = d * _e2_H(1 + _e2_H(radial / 3))\n",
                  "            birth = d * _e2_H(1 + _e2_H(radial * _BR_X3))   # opus-engineer BR: the exchange part of the birth coefficient times the lead's dial\n", 1))
    parts.append('# BEGIN OPUS-ENGINEER BIRTH DIAL (the lead\'s e27 arm br: birth coefficient d (1 + x radial / 3); _BR_X3 = x / 3)\n'
                 '_BR_X3 = %r\n# END OPUS-ENGINEER BIRTH DIAL' % (float(br[0][2:]) / 3))
ds = [x for x in levers if x.startswith('DS')]
if ds:
    assert len(ds) == 1
    edits.append(("                                  head_b=P['head_b'][l], gain=1 + 0.25 * fnp.tanh(P['gain'][l:l + 1]))\n",
                  "                                  head_b=P['head_b'][l], gain=(1 + 0.25 * fnp.tanh(P['gain'][l:l + 1])) * _DS_SCALE)   # opus-engineer DS: the learned mean correction times a global scale\n", 1))
    parts.append('# BEGIN OPUS-ENGINEER MEAN-CORRECTION SCALE (a22, 2026-10-02; an ERROR lever: the learned mean correction of layers 1-15 times one\n'
                 '# number; it rides on the gain constant of gnn_step, formed once per layer and process)\n'
                 '_DS_SCALE = %r\n# END OPUS-ENGINEER MEAN-CORRECTION SCALE' % float(ds[0][2:]))
gn = [x for x in levers if x.startswith('GN')]
if gn:
    import re
    knobs = {}
    for x in gn:
        match = re.fullmatch(r'GN(iso|k21n|k31)(-?[0-9.]+)', x)
        assert match and match.group(1) not in knobs, x
        knobs[match.group(1)] = float(match.group(2))
    assert PARENT.read_text().count("_GN_KNOBS = {'iso': -0.094, 'k21n': -0.028, 'k31': -0.212, 'edge': 0.164, 'rc': -0.08}\n") == 1
    parts.append('# BEGIN OPUS-ENGINEER COVARIANCE-OUTPUT KNOBS (values of a refit; read at call time by _gn_dcov and at setup by _g5_setup)\n'
                 '_GN_KNOBS = dict(_GN_KNOBS, %s)\n# END OPUS-ENGINEER COVARIANCE-OUTPUT KNOBS' % ', '.join('%s=%r' % kv for kv in sorted(knobs.items())))
ynt = [x for x in levers if x.startswith('YNT')]
if ynt:
    assert len(ynt) == 1 and not [x for x in levers if x.startswith('Zynt')]
    edits += [("    negative = (ctx['a'] < -2.5) & (~ctx['live'])\n", "    negative = (ctx['a'] < -_YNT_Z) & (~ctx['live'])\n", 1),
              ("    score = fnp.where(negative, ctx['a'] + 2.5, 0.0)\n", "    score = fnp.where(negative, ctx['a'] + _YNT_Z, 0.0)\n", 1)]
    parts.append('# BEGIN OPUS-ENGINEER YOUNG DEAD-TAIL THRESHOLD (a22, 2026-10-02; an ERROR lever: the young columns of dead rows with\n'
                 '# -_YNT_Z < alpha < -2.5 are pushed again instead of being replaced by the spherical trace; the parent cut at 2.5)\n'
                 '_YNT_Z = %r\n# END OPUS-ENGINEER YOUNG DEAD-TAIL THRESHOLD' % float(ynt[0][3:]))
if 'L3' in levers:
    parts.append((E/'a21_load3/fragment_load3.py').read_text().strip('\n'))
if 'L3o' in levers:
    assert 'L3' in levers
    parts.append((E/'a21_load3/fragment_load3o.py').read_text().strip('\n'))
bvp = [x for x in levers if x[0] == 'V']
if bvp:
    wall = bvp[0][1:] or '0.45'
    float(wall)
    frag = (E/'a11_bare_views/fragment_bvp.py').read_text()
    kind = {'A': 0, 'B': 1, 'C': 2, 'acc': 3}
    fam = '(' + ', '.join('(%d, %d, range(%d, %d), range(%d, %d))' % (kind[s], L, rs.start, rs.stop, cs.start, cs.stop) for s, L, rs, cs in FAMILIES) + ',)'
    old = "_BVP_FAMILIES = ()                          # ((kind, L, r range, c range), ...), most valuable first; set by the builder"
    assert frag.count(old) == 1 and frag.count('_BVP_WALL_S = 0.45 ') == 1
    parts.append(frag.replace(old, "_BVP_FAMILIES = " + fam + "   # (kind, L, leaf rows, leaf cols), most valuable first")
                 .replace('_BVP_WALL_S = 0.45 ', '_BVP_WALL_S = %s ' % wall).strip('\n'))
cuts = [x[1:] for x in levers if x[0] == 'Z']
if cuts:
    names = dict(ott='_OTT_Z', neg='_OTT_NEG_Z', ynt='_YNT_Z', dead='_DEADCUT_Z', l15_='_L15_ROWCUT_Z')
    lines = ['# BEGIN OPUS-ENGINEER AUDIT OVERRIDES (row-cut thresholds; not a candidate)']
    for cut in cuts:
        kind = next(k for k in sorted(names, key=len, reverse=True) if cut.startswith(k))
        lines.append('%s = %r' % (names[kind], float(cut[len(kind):])))
        if kind == 'ynt':
            edits += [("    negative = (ctx['a'] < -2.5) & (~ctx['live'])\n", "    negative = (ctx['a'] < -_YNT_Z) & (~ctx['live'])\n", 1),
                      ("    score = fnp.where(negative, ctx['a'] + 2.5, 0.0)\n", "    score = fnp.where(negative, ctx['a'] + _YNT_Z, 0.0)\n", 1)]
    parts.append('\n'.join(lines + ['# END OPUS-ENGINEER AUDIT OVERRIDES']))
for name in [x[1:] for x in levers if x[0] == 'P']:
    # the list fragment also names the number of listed keys fused as they are bound (the parent's 128), so that it can be tuned
    edits.append(("(kind, L, rows, cols) in _PB_SINGLE[:128])) else None\n", "(kind, L, rows, cols) in _PB_SINGLE[:_PB_EAGER])) else None\n", 1))
    parts.append((E/'a13_prebind'/('prebind_%s.py' % name)).read_text().strip('\n'))
so2 = [x for x in levers if x.startswith('SO2m')]
if so2:
    import re
    assert len(so2) == 1
    m_so2 = re.fullmatch(r'SO2m([0-9.]+)(?:r([0-9.]+))?(?:t([0-9.]+)(d)?(?:k([01]))?(?:p([1-6]+))?)?(?:f([0-9.]+))?', so2[0])
    assert m_so2, so2
    so2_t, so2_f = float(m_so2.group(3) or 0.0), float(m_so2.group(7) or 0.0)
    if not (so2_t or so2_f):   # the mean term alone: v1, byte for byte the builds of 18:00Z
        frag = (E/'a41_base/fragment_so2_v1.py').read_text()
        assert frag.count('_SO2_MEAN = 0.0 ') == 1 and frag.count('_SO2_RAW2 = None ') == 1
        parts.append(frag.replace('_SO2_MEAN = 0.0 ', '_SO2_MEAN = %r ' % float(m_so2.group(1)))
                     .replace('_SO2_RAW2 = None ', '_SO2_RAW2 = %r ' % (float(m_so2.group(2)) if m_so2.group(2) else None)).strip('\n'))
    else:                      # v2: + the second-order part of the third-cumulant target (t, k = highest power of rho, p = parts as digits 1-6 = a b q c d e) and the final layer's mean term (f)
        frag = (E/'a41_base/fragment_so2_v2.py').read_text()
        subs = [('_SO2_MEAN = 0.0 ', '_SO2_MEAN = %r ' % float(m_so2.group(1))), ('_SO2_RAW2 = None ', '_SO2_RAW2 = %r ' % (float(m_so2.group(2)) if m_so2.group(2) else None)),
                ('_SO2_TGT = 0.0 ', '_SO2_TGT = %r ' % so2_t), ('_SO2_KMAX = 1 ', '_SO2_KMAX = %r ' % int(m_so2.group(5) or 1)),
                ("_SO2_PARTS = 'abqcde' ", '_SO2_PARTS = %r ' % ''.join('abqcde'[int(ch) - 1] for ch in (m_so2.group(6) or '123456'))),
                ('_SO2_TDIAG = False ', '_SO2_TDIAG = %r ' % bool(m_so2.group(4))), ('_SO2_FINAL = 0.0 ', '_SO2_FINAL = %r ' % so2_f)]
        for old_s, new_s in subs:
            assert frag.count(old_s) == 1, old_s
            frag = frag.replace(old_s, new_s)
        parts.append(frag.strip('\n'))
        if so2_f:
            edits.append(("                m = m + _e2_H(final_hook(l, m, aux, _e2_H(W[l])))\n",
                          "                m = m + _e2_H(_so2_final(_e2_H(final_hook(l, m, aux, _e2_H(W[l]))), aux))   # opus-engineer so2: the final layer's second-order mean term, added to the learned correction\n", 1))
bsk = [x for x in levers if x.startswith('BSK')]
if bsk:
    assert len(bsk) == 1
    edits.append(("            birth = d * _e2_H(1 + _e2_H(radial / 3))\n",
                  "            birth = d * _e2_H(1 + _e2_H(radial / 3))\n"
                  "            birth = _bsk(birth, aux)   # opus-engineer bsk: the kink density to first order in the unit's skewness\n", 1))
    frag = (E/'a41_base/fragment_bsk_v1.py').read_text()
    assert frag.count('_BSK = 0.0\n') == 1
    parts.append(frag.replace('_BSK = 0.0\n', '_BSK = %r\n' % float(bsk[0][3:])).strip('\n'))
bk3 = [x for x in levers if x.startswith('BK3')]
if bk3:
    import re
    assert len(bk3) == 1
    m_bk3 = re.fullmatch(r'BK3([pr])([0-9.]+),([0-9.]+)(?:k([0-9.]+))?', bk3[0])
    assert m_bk3, bk3
    assert m_bk3.group(1) == 'r', "bk3 level 'pair' as written changes the K21 plane for EVERY consumer (features, edge, covariance knob, birth update); the reference changes it for the pair map only: use BK3r, or give the pair map its own plane first"
    if m_bk3.group(1) == 'p':
        k21_line = "            _ps_K2132 = (_ps_cp32 + (_plane_star32 if _plane_star32 is not None else _ps32(star_pre)) + fnp.outer(_ps32(_ps_bv), _ps_bpre32)) + fnp.einsum('ij,i->ij', _ps_pre32(Bpre), _ps32(2 * bpre), optimize=False)\n"
        edits.append((k21_line,
                      "            _bk3 = _bk3_fold(l, _ps_cp32 + (_plane_star32 if _plane_star32 is not None else _ps32(star_pre)), _ps_bv, mu, vv, bpre, _plane_cpdiag64, star_pre)   # opus-engineer bk3: the collective mode times the specific third cumulant, folded into the K21 plane\n"
                      "            _ps_K2132 = (_bk3[0] + fnp.outer(_ps32(_bk3[1]), _ps_bpre32)) + fnp.einsum('ij,i->ij', _ps_pre32(Bpre), _ps32(2 * bpre), optimize=False)\n", 1))
    if m_bk3.group(1) == 'r':   # the loop hands its whole frame to the hook and holds it; this term needs five names: pass those, so that no dead array of the layer stays referenced through the next layer's peak
        edits.append(("extra31=_e2_H(_t31_extra(l, _e2_H(locals()))))\n",
                      "extra31=_e2_H(_t31_extra(l, dict(bpre=bpre, _plane_star32=_plane_star32, _ps_cp32=_ps_cp32, star_pre=star_pre, _plane_cpdiag64=_plane_cpdiag64))))   # opus-engineer bk3: the five arrays the extra plane is made of\n", 1))
    frag = (E/'a41_base/fragment_bk3_v1.py').read_text()
    subs = [('_BK3_K31 = 0.0 ', '_BK3_K31 = %r ' % float(m_bk3.group(4) or 1.0)), ('_BK3_X = (3.2, 10.0) ', '_BK3_X = %r ' % ((float(m_bk3.group(2)), float(m_bk3.group(3))),)),
            ("_BK3_LEVEL = 'pair'\n", "_BK3_LEVEL = %r\n" % ('pair' if m_bk3.group(1) == 'p' else 'r31'))]
    for old_s, new_s in subs:
        assert frag.count(old_s) == 1, old_s
        frag = frag.replace(old_s, new_s)
    parts.append(frag.strip('\n'))
gsk = [x for x in levers if x.startswith('GSK')]
if gsk:
    import re
    assert len(gsk) == 1
    m_gsk = re.fullmatch(r'GSK([0-9.]+)(?:k([0-9.]+))?(?:b([0-9.]+))?', gsk[0])
    assert m_gsk, gsk
    b_line = '            b = _e2_H(g * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / _S19_BSRC_DIV) if _FULL_LOWER_MOMENTS else _e2_H(_e2_H(g * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / _S19_BSRC_DIV)) * alive'
    edits += [
        ("            d = _e2_pair['d'] if _e2_pair is not None and _e2_H(_e2_pair['aux']) is aux else aux['pdf'] / aux['sigma']\n",
         "            d = _e2_pair['d'] if _e2_pair is not None and _e2_H(_e2_pair['aux']) is aux else aux['pdf'] / aux['sigma']\n"
         "            gq = _gq(g, d, aux, bpre)   # opus-engineer gsk: the gate that carries the third-order content, with the mean gate's first-order corrections\n", 1),
        (b_line + "\n", b_line.replace('g * bpre', 'gq * bpre') + "\n", 1),
        ("            _ps_g2 = g ** 2\n"
         "            _ps_inherited_diag = fnp.einsum('i,i,i->i', _ps_g2, g, _plane_cpdiag64, optimize=False)\n"
         "            inherited = fnp.einsum('i,j,ij->ij', _ps32(_ps_g2), _ps32(g), _ps_cp32, optimize=False)\n",
         "            _ps_g2 = gq ** 2\n"
         "            _ps_inherited_diag = fnp.einsum('i,i,i->i', _ps_g2, gq, _plane_cpdiag64, optimize=False)\n"
         "            inherited = fnp.einsum('i,j,ij->ij', _ps32(_ps_g2), _ps32(gq), _ps_cp32, optimize=False)\n", 1),
        ("            _e2_ga = _e2_H(g.astype(fnp.float32)) * alive\n", "            _e2_ga = _e2_H(gq.astype(fnp.float32)) * alive\n", 1),
        ("            _E2_LAYER.update(ga_gate=g, ga_alive=alive, ga=_e2_ga)\n            _shared_old384_commit(l, g, alive, selected, shared_old)\n",
         "            _E2_LAYER.update(ga_gate=gq, ga_alive=alive, ga=_e2_ga)\n            _shared_old384_commit(l, gq, alive, selected, shared_old)\n", 1),
    ]
    if f4c:   # the exact-term mixture's anchor is made of the gated columns: it takes the gate those columns were gated with
        edits.append(("_f4c_step(l, right if len(lam) else None, g, d, live_indices", "_f4c_step(l, right if len(lam) else None, gq, d, live_indices", 1))
    frag = (E/'a41_base/fragment_gsk_v1.py').read_text()
    subs = [('_GSK = 0.0 ', '_GSK = %r ' % float(m_gsk.group(1))), ('_GK4 = 0.0 ', '_GK4 = %r ' % float(m_gsk.group(2) or 0.0)), ('_GBK = 0.0 ', '_GBK = %r ' % float(m_gsk.group(3) or 0.0))]
    for old_s, new_s in subs:
        assert frag.count(old_s) == 1, old_s
        frag = frag.replace(old_s, new_s)
    parts.append(frag.strip('\n'))
if 'CS' in levers:
    parts.append((E/'a40_setup/fragment_cs_v1.py').read_text().strip('\n'))
ro = [x for x in levers if x.startswith('SRO@')]
if ro:   # the trainer's state readouts (o2) carried b and (o3) radial (a43_readout/fragment_ro_v1.py): one loop edit at the end of the layer, a wrapper of gnn_step
    assert len(ro) == 1
    ro_spec = __import__('json').loads((E/('a43_readout/weights/%s.json' % ro[0][4:])).read_text())
    for key in ('b', 'rad'):
        assert ro_spec.get(key) is None or (len(ro_spec[key]) == 16 and all(len(r) == 64 for r in ro_spec[key])), key
    assert ro_spec.get('b') is not None or ro_spec.get('rad') is not None
    edits.append(("            radial = radial_new\n",
                  "            b = _ro_b(l, b, C)   # opus-engineer ro: the carried b with the state readout (o2)\n"
                  "            radial = _ro_radial(l, radial_new)   # opus-engineer ro: the radial with the state readout (o3)\n", 1))
    ro_frag = (E/'a43_readout/fragment_ro_v1.py').read_text()
    assert ro_frag.count('_RO_B = None ') == 1 and ro_frag.count('_RO_RAD = None ') == 1
    fmt = lambda a: None if a is None else tuple(tuple(float(x) for x in row) for row in a)
    parts.append(ro_frag.replace('_RO_B = None ', '_RO_B = %r ' % (fmt(ro_spec.get('b')),)).replace('_RO_RAD = None ', '_RO_RAD = %r ' % (fmt(ro_spec.get('rad')),)).strip('\n'))
hcf = [x for x in levers if x.startswith('HCF@')]
if hcf:   # the final layer's head columns (a42_head/fragment_hcf_v1.py): one line of final_delta, a wrapper of the final layer's K4L value
    assert len(hcf) == 1 and k4l, 'the final-layer columns need the K4L lever (its final-layer value)'
    assert any(x.startswith('HC@') for x in levers), 'the final-layer columns ride on the HC lever\'s first-layer bias (gnn_step at layer 15)'
    hcf_spec = __import__('json').loads((E/('a42_head/weights/%s.json' % hcf[0][4:])).read_text())
    hcf_columns = hcf_spec['columns']
    hcf_v2 = any(c.startswith('f15_') for c in hcf_columns)
    assert hcf_columns and all(c in (('f15_diag', 'f15_k31l_diag', 'f15_k4_local') if hcf_v2 else ('k4_local15',)) for c in hcf_columns) and len(set(hcf_columns)) == len(hcf_columns), hcf_columns
    assert len(hcf_spec['u']) == len(hcf_columns) and all(len(u) == 128 for u in hcf_spec['u'])
    hcf_frag = (E/('a42_head/fragment_hcf_v3.py' if hcf_v2 else 'a42_head/fragment_hcf_v1.py')).read_text()   # v3: column 18 on ages 1-2 at strength 1 (the trainer's definition)
    assert hcf_frag.count('_HCF_COLUMNS = () ') == 1 and hcf_frag.count('_HCF_U = None ') == 1
    parts.append(hcf_frag.replace('_HCF_COLUMNS = () ', '_HCF_COLUMNS = %r ' % (tuple(hcf_columns),))
                 .replace('_HCF_U = None ', '_HCF_U = %r ' % (tuple(tuple(float(x) for x in u) for u in hcf_spec['u']),)).strip('\n'))
dcv = [x for x in levers if x.startswith('DCV')]
if dcv:   # the covariance sandwich with the dead units of the layer below cut (a44_cov/fragment_dcv_v1.py): two call-site edits
    assert len(dcv) == 1 and ss, 'the cut sandwich copies the SS fragment\'s star lines (_ss_factors)'
    edits += [
        ("                        Cpre, _bb_star_pre, _plane_cov32, _plane_star32 = _flms_cov_star(w32, C, star, retain32=True)\n",
         "                        Cpre, _bb_star_pre, _plane_cov32, _plane_star32 = _dc_cov_star(w32, C, star, response_gate, gate_floor, retain32=True)   # opus-engineer dcv: the units of the layer below under the gate floor cut\n", 1),
        ("                v, active_v = (_flms_covariance_diagonal(w32, C) if _FULL_LOWER_MOMENTS\n",
         "                v, active_v = (_dc_cov_diagonal(w32, C, response_gate, gate_floor) if _FULL_LOWER_MOMENTS   # opus-engineer dcv: the same cut\n", 1),
    ]
    import re
    m_dcv = re.fullmatch(r'DCV([0-9]*)(?:f([0-9.e-]+))?([gpt])?', dcv[0])
    assert m_dcv, dcv
    dcv_frag = (E/('a44_cov/fragment_dcv_v3.py' if m_dcv.group(3) else 'a44_cov/fragment_dcv_v2.py' if m_dcv.group(2) else 'a44_cov/fragment_dcv_v1.py')).read_text()
    if m_dcv.group(3):   # the second product's kernel at a cut roster: g generic, p padded specialised product, t tiles at the inner size
        assert dcv_frag.count("_DCV_SYM = 'generic'\n") == 1
        dcv_frag = dcv_frag.replace("_DCV_SYM = 'generic'\n", "_DCV_SYM = %r\n" % {'g': 'generic', 'p': 'pad', 't': 'tiles'}[m_dcv.group(3)])
    assert dcv_frag.count('_DCV_ROUND = 64\n') == 1
    dcv_frag = dcv_frag.replace('_DCV_ROUND = 64\n', '_DCV_ROUND = %d\n' % int(m_dcv.group(1) or 64))
    if m_dcv.group(2):
        assert dcv_frag.count('_DCV_FLOOR = None\n') == 1
        dcv_frag = dcv_frag.replace('_DCV_FLOOR = None\n', '_DCV_FLOOR = %r\n' % float(m_dcv.group(2)))
    parts.append(dcv_frag.strip('\n'))
l15 = [x for x in levers if x.startswith('L15@')]
if l15:   # layer 15's own copy of the learned correction (a45_l15/fragment_l15_v1.py): four lines of gnn_step, a wrapper of _e2_gnn_layer
    assert len(l15) == 1
    enc0_line = ("    h = silu(_e2_H(_e2_H(k['H'] @ k['enc0_wT']) + _e2_H(_hc_bias(l, P['enc0_b']))))   # opus-engineer hc: enc0_b + z u' when the layer's read made the new column\n"
                 if any(x.startswith('HC@') for x in levers) else "    h = silu(_e2_H(_e2_H(k['H'] @ k['enc0_wT']) + P['enc0_b']))\n")
    edits += [
        ("    M = _e2_messages(k, W, state)\n", "    M = _e2_messages(_l15_msgk(k, l), W, state)   # opus-engineer l15: layer 15's own message weights when given\n", 1),
        (enc0_line, "    _l15_e0w, _l15_e0b, _l15_e2w, _l15_e2b, _l15_e4w, _l15_e4b = _l15_enc(k, l, P)   # opus-engineer l15: layer 15's own encoder when given, else the shared one\n"
                    + enc0_line.replace("k['enc0_wT']", "_l15_e0w").replace("P['enc0_b']", "_l15_e0b"), 1),
        ("    h = silu(_e2_H(_e2_H(h @ k['enc2_wT']) + P['enc2_b']))\n", "    h = silu(_e2_H(_e2_H(h @ _l15_e2w) + _l15_e2b))\n", 1),
        ("    state = _e2_H(h @ k['enc4_wT']) + P['enc4_b']\n", "    state = _e2_H(h @ _l15_e4w) + _l15_e4b\n", 1),
    ]
    l15_frag = (E/'a45_l15/fragment_l15_v1.py').read_text()
    assert l15_frag.count('_L15_W = None\n') == 1
    if l15[0] != 'L15@off':
        import base64, zlib, numpy
        l15_npz = numpy.load(E/('a45_l15/weights/%s.npz' % l15[0][4:]))
        shapes = dict(enc0_w=(128, 126), enc0_b=(128,), enc2_w=(128, 128), enc2_b=(128,), enc4_w=(64, 128), enc4_b=(64,), msg_w=(48, 64), msg_b=(48,),
                      centers=(30,), scales=(30,), rms=(), head=(64,), head_b=(), gain=(1,))
        l15_lit = {}
        for key in l15_npz.files:
            arr = numpy.asarray(l15_npz[key], dtype=numpy.float32)
            assert key in shapes and tuple(arr.shape) == shapes[key], (key, arr.shape)
            assert numpy.all(numpy.isfinite(arr)), key
            l15_lit[key] = (tuple(arr.shape), base64.b64encode(zlib.compress(arr.tobytes(), 9)).decode())
        assert l15_lit
        l15_frag = l15_frag.replace('_L15_W = None\n', '_L15_W = %r\n' % (l15_lit,))
    parts.append(l15_frag.strip('\n'))
kc = [x for x in levers if x.startswith('KC@')]
if kc:   # the trainer's kappa_C (a47_kc/fragment_kc_v1.py): the loop's rank-2 covariance update after the mean correction takes rc_l - kc_l
    assert len(kc) == 1
    kc_spec = __import__('json').loads((E/('a47_kc/kc/%s.json' % kc[0][3:])).read_text())['kc']
    kc_values = {int(k): float(v) for k, v in kc_spec.items()}
    assert all(1 <= k <= 14 for k in kc_values), kc_values
    edits.append(("                if _GN_RC.get(l):\n"
                  "                    _ps_Cdiag = _ps_Cdiag - _GN_RC[l] * (_ps_Cdiag0 - _ps_Cdiag)\n"
                  "                    C = C - _ps32(_GN_RC[l]) * (_ps_C0 - C)\n",
                  "                _gn_rc = _GN_RC.get(l, 0.0) - _GN_KC.get(l, 0.0)   # opus-engineer kc: the trainer's kappa_C adds kc_l x the rank-2 update back\n"
                  "                if _gn_rc:\n"
                  "                    _ps_Cdiag = _ps_Cdiag - _gn_rc * (_ps_Cdiag0 - _ps_Cdiag)\n"
                  "                    C = C - _ps32(_gn_rc) * (_ps_C0 - C)\n", 1))
    kc_frag = (E/'a47_kc/fragment_kc_v1.py').read_text()
    assert kc_frag.count('_GN_KC = {}\n') == 1
    parts.append(kc_frag.replace('_GN_KC = {}\n', '_GN_KC = %r\n' % ({k: v for k, v in sorted(kc_values.items()) if v},)).strip('\n'))
band = [x for x in levers if x.startswith('BAND@')]
if band:   # per-band head parameters (a49_band/fragment_band_v1.py): the outermost wrapper of _hc_bias (after the HC and HCF fragments)
    assert len(band) == 1 and any(x.startswith('HC@') for x in levers), 'the per-band parameters ride on the HC lever'
    band_spec = __import__('json').loads((E/('a49_band/weights/%s.json' % band[0][5:])).read_text())
    edges = tuple(tuple(int(v) for v in e) for e in band_spec.get('edges', [[1, 9], [10, 14], [15, 15]]))
    enc = band_spec.get('enc0_b')
    ub = band_spec.get('u')
    if ub is not None:
        assert band_spec.get('columns') == hc_spec.get('columns', ['diag']), 'the per-band weights must name the HC lever\'s columns in its order'
        assert len(ub) == len(edges) and all(m is None or (len(m) == len(band_spec['columns']) and all(len(r) == 128 for r in m)) for m in ub)
    if enc is not None:
        assert len(enc) == len(edges) and all(v is None or len(v) == 128 for v in enc)
    fmt_vec = lambda v: None if v is None else tuple(float(x) for x in v)
    band_frag = (E/'a49_band/fragment_band_v1.py').read_text()
    for old_s, new_s in (('_BAND_EDGES = ((1, 9), (10, 14), (15, 15))\n', '_BAND_EDGES = %r\n' % (edges,)),
                         ('_ENC0_B_BAND = None\n', '_ENC0_B_BAND = %r\n' % (None if enc is None else tuple(fmt_vec(v) for v in enc),)),
                         ('_HC_U_BAND = None\n', '_HC_U_BAND = %r\n' % (None if ub is None else tuple(None if m is None else tuple(fmt_vec(r) for r in m) for m in ub),))):
        assert band_frag.count(old_s) == 1, old_s
        band_frag = band_frag.replace(old_s, new_s)
    parts.append(band_frag.strip('\n'))
trk = [x for x in levers if x.startswith('TRK')]
if trk:   # the old tier's rank at the three joins (the literal of the R320 router)
    assert len(trk) == 1
    ranks = [int(x) for x in trk[0][3:].split(',')]
    assert len(ranks) == 3 and all(r % 64 == 0 and 256 <= r <= 512 for r in ranks), ranks   # above 384: the join's sketch and QR take the rank; the old tier's checks need >= 384 live rows
    edits.append(("_R320_RULE = {7: 384, 10: 320, 13: 384}   # join layer",
                  "_R320_RULE = {7: %d, 10: %d, 13: %d}   # opus-engineer TRK: the old tier's rank by join; join layer" % tuple(ranks), 1))
pld = [x for x in levers if x.startswith('DLY@')]
if pld:   # per-layer values of global dials (a52_pld/fragment_pld_v1.py): Python lookups at the literals' use sites
    assert len(pld) == 1
    pld_spec = __import__('json').loads((E/('a52_pld/values/%s.json' % pld[0][4:])).read_text())
    pld_values = {name: {int(k): float(v) for k, v in pld_spec[name].items()} for name in ('bscale', 'X1', 'iso', 'k4la1', 'scarry') if pld_spec.get(name)}
    assert set(pld_spec) <= {'bscale', 'X1', 'iso', 'k4la1', 'scarry', 'source', 'note'}, sorted(pld_spec)
    if 'bscale' in pld_values:
        edits.append(("                b = b * _S19_BSCALE\n", "                b = b * _pld('bscale', l, _S19_BSCALE)   # opus-engineer pld: per-layer bscale\n", 1))
    if 'X1' in pld_values:
        edits.append(("_BD_J1_PHYS * (1.0 - _j1m)", "_pld('X1', _PHYS_LAYER[0], _BD_J1_PHYS) * (1.0 - _j1m)", 4))
    if 'iso' in pld_values:
        edits.append(("    th = _GN_KNOBS\n", "    th = _pld_knobs(l)   # opus-engineer pld: per-layer iso\n", 2))
    if 'scarry' in pld_values:   # 2026-10-11: s_carry by layer (fragment v3): the loop's carried lam here, F4c's step and the pair law's
        # per-layer constants in their appended fragments (each text once), F4c's birth mix by the v3 wrapper
        edits.append(("lam * _S19_LAM[0], _s19_new, -_s19_new", "lam * _pld('scarry', l, _S19_LAM[0]), _s19_new, -_s19_new", 1))
        for old_s, new_s in (("        lam2 = lam2 * _S19_LAM[0]", "        lam2 = lam2 * _pld('scarry', l, _S19_LAM[0])"),
                             ("[{'#aV': 2.0 * kap[l + 1] / carry, '#aG': 2.0 * kap[l + 1] / (carry * kap[l])} for l in layers]",
                              "[{'#aV': 2.0 * kap[l + 1] / _pldc(l, carry), '#aG': 2.0 * kap[l + 1] / (_pldc(l, carry) * kap[l])} for l in layers]"),
                             ("C['aV'] = [2.0 * kap[l + 1] / carry for l in layers]", "C['aV'] = [2.0 * kap[l + 1] / _pldc(l, carry) for l in layers]"),
                             ("[{'#iaV': carry / (2.0 * kap[l + 1]), '#iaG': carry * kap[l] / (2.0 * kap[l + 1]),",
                              "[{'#iaV': _pldc(l, carry) / (2.0 * kap[l + 1]), '#iaG': _pldc(l, carry) * kap[l] / (2.0 * kap[l + 1]),"),
                             ("def _np_tables(n):\n", "def _pldc(l, carry):\n    # opus-engineer pld v3: s_carry of layer l once the per-layer values exist (pld v3 rebuilds _NP_TABLES then: the\n"
                              "    # import-time build below runs before _PLD is defined)\n    d = globals().get('_PLD', {}).get('scarry')\n"
                              "    return d[l] if d and l in d else carry\n\n\ndef _np_tables(n):\n")):
            hits = [i for i, part in enumerate(parts) if old_s in part]
            assert len(hits) == 1 and parts[hits[0]].count(old_s) == 1, ('scarry edit', old_s[:60], hits)
            parts[hits[0]] = parts[hits[0]].replace(old_s, new_s)
    pld_frag = (E/('a52_pld/fragment_pld_v3.py' if 'scarry' in pld_values else 'a52_pld/fragment_pld_v2.py')).read_text()   # v2: K4L v6 and v7 (v1: v6 only); v3: + scarry
    assert pld_frag.count('_PLD = {}\n') == 1
    parts.append(pld_frag.replace('_PLD = {}\n', '_PLD = %r\n' % (pld_values,)).strip('\n'))
for name in [x[1:] for x in levers if x[0] == 'R']:   # a release-lane fragment, appended last exactly as release/build_variant.py appends it
    import hashlib
    path = E.parent/'release/variants/fragments'/(name + '.py')
    body = path.read_text()
    parts.append("# BEGIN OPUS-RELEASE FRAGMENT %s (sha256 %s)\n%s\n# END OPUS-RELEASE FRAGMENT %s" % (name, hashlib.sha256(path.read_bytes()).hexdigest()[:12], body.rstrip(), name))
if 'XRQ' in levers or 'XRQ2' in levers:   # request diet (a56_req/fragment_rqd_v1.py; XRQ2 = v2 + a per-layer float32 conversion cache)
    parts.append((E/('a56_req/fragment_rqd_v2.py' if 'XRQ2' in levers else 'a56_req/fragment_rqd_v1.py')).read_text().strip('\n'))
hz7 = [x for x in levers if x.startswith('HZ7')]
if hz7:   # 7-level strip pushes (a57_hz7/fragment_hz7_v1.py): HZ7 = the rule's 3% gain floor, HZ7g<x> = a gain floor of x (0.0 = every gain)
    assert len(hz7) == 1
    gain = 0.03 if hz7[0] in ('HZ7', 'HZ7P', 'HZ7R') else float(hz7[0][5:] if hz7[0][:5] in ('HZ7Pg', 'HZ7Rg') else hz7[0][4:])
    hz7_frag = (E/('a57_hz7/fragment_hz7_v4.py' if hz7[0].startswith('HZ7R') else 'a57_hz7/fragment_hz7_v3.py' if hz7[0].startswith('HZ7P') else 'a57_hz7/fragment_hz7_v1.py')).read_text()   # HZ7P = v3 (+ strip views bound in setup), HZ7R = v4 (full packets at 7 levels, the remainder by the parent)
    assert hz7_frag.count('_HZ7_GAIN = 0.03\n') == 1
    parts.append(hz7_frag.replace('_HZ7_GAIN = 0.03\n', '_HZ7_GAIN = %r\n' % gain).strip('\n'))
if 'XRQS' in levers:   # persistent scratch (a56_req/fragment_rqs_v1.py), after XRQ2 and HZ7R
    parts.append((E/'a56_req/fragment_rqs_v1.py').read_text().strip('\n'))
hzd = [x for x in levers if x in ('HZD', 'HZD2', 'HZD3')]
if hzd:   # direct strip loads (a59_direct/fragment_hzd_v1.py: k-split reads; v2: + strip products; v3: v2 + a bounded view cache)
    assert len(hzd) == 1
    parts.append((E/('a59_direct/fragment_hzd_%s.py' % {'HZD': 'v1', 'HZD2': 'v2', 'HZD3': 'v3'}[hzd[0]])).read_text().strip('\n'))
if 'XSDM' in levers:   # the dead-cut sandwich at its inner size (a60_psdm/fragment_psdm_v1.py), after XRQS
    assert 'XRQS' in levers and any(x.startswith('DCV') and x.endswith('p') for x in levers), 'XSDM needs XRQS and the padded dead cut'
    parts.append((E/'a60_psdm/fragment_psdm_v1.py').read_text().strip('\n'))
u15 = [x for x in levers if x.startswith('U15')]
if u15:   # the final layer's masked inputs 13-15 and 26 from the full C_pre15 (a55_u15/fragment_u15_v1.py); U15a = all four, U15<p>.<p>.. = those
    assert len(u15) == 1 and any(x.startswith('DCV') for x in levers), 'U15 wraps the dead-cut diagonal (DCV lever needed)'
    pos = (13, 14, 15, 26) if u15[0] == 'U15a' else tuple(int(t) for t in u15[0][3:].split('.'))
    assert pos and set(pos) <= {13, 14, 15, 26}, pos
    edits += [("    ls = fnp.log(s); ms = m/s; vs = v*s; vsq = v*v\n",
               "    ls = fnp.log(s); ms = m/s; vs = v*s; vsq = v*v\n    _u15 = _u15_columns(n, aux, zero)   # opus-engineer u15: features 13-15, 26 from the full C_pre15\n", 1),
              ("zero, zero, zero, _e2_H(ws[0])", "_u15[13], _u15[14], _u15[15], _e2_H(ws[0])", 1),
              ("_e2_H(aux['post_k4'] / vsq), zero, zero, zero, zero, zero, zero, zero, _e2_H(ones",
               "_e2_H(aux['post_k4'] / vsq), zero, zero, zero, zero, _u15[26], zero, zero, _e2_H(ones", 1),
              ("mask=fnp.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=f32)",
               "mask=fnp.asarray([1 if (i not in (13, 14, 15, 22, 23, 24, 25, 26, 27, 28) or i in _U15_POS) else 0 for i in range(30)], dtype=f32)", 1)]
    u15_frag = (E/'a55_u15/fragment_u15_v1.py').read_text()
    assert u15_frag.count('_U15_POS = (13, 14, 15, 26)\n') == 1
    parts.append(u15_frag.replace('_U15_POS = (13, 14, 15, 26)\n', '_U15_POS = %r\n' % (pos,)).strip('\n'))
ovr = [x for x in levers if x.startswith('OVR@')]
if ovr:   # literal overrides, appended after everything else (the last assignment of a module-level name wins at call time)
    assert len(ovr) == 1
    import json as _json
    ovr_spec = _json.loads((E/('a50_ovr/%s.json' % ovr[0][4:])).read_text())
    body = PARENT.read_text() + '\n'.join(parts)
    lines = ['# BEGIN OPUS-ENGINEER LITERAL OVERRIDES (%s): values only, read at call time' % ovr[0]]
    for key, value in ovr_spec.items():
        if key.startswith('_comment'):
            continue
        assert __import__('re').search(r'(?m)^%s = ' % __import__('re').escape(key), body), 'no literal %s in the file' % key
        lines.append('%s = %r' % (key, value))
    lines.append('# END OPUS-ENGINEER LITERAL OVERRIDES')
    parts.append('\n'.join(lines))
build(PARENT, S/('candidate_%s%s.py' % (spec.replace('+', '_'), TAG)), edits=edits, append='\n\n\n'.join(parts), expect_parent_sha=PARENT_SHA)
