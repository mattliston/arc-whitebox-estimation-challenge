# otr_m_sc4 (scout-card, 2026-10-12): = otr_m_sc3.py + KB: the carried b vector moved against the learned mean correction, b <- b - KB (m - host)
# at layers 0-14 (after the card's bscale; the K21 b-form's share of recentring the third cumulant by the head's d, the analogue of kappa_C for C).
# KB None or 0 = otr_m_sc3 bit for bit (set per request by worker_sc4.py from the field 'kb').
# otr_m_sc3 (scout-card, 2026-10-12): = otr_m_sc2.py with K4L's age limit 3 -> 4 and the family 'k4la4' (shifts k4la3 = k4la4 = -1 reproduce otr_m_v5).
# otr_m_sc2 (scout-card, 2026-10-12): = otr_m_sc1.py + BFIXG (the gain b-fix gamma, the card knob 'bfix_gamma', S2 value 0.1): None = the card's own value;
# a number replaces it for the whole network (set per request by worker_sc2.py from the field 'bfixg'). Lines of sc1 follow.
# otr_m_sc1 (scout-card, 2026-10-12): = trainer/otr_m_v5.py (sha 7553b39e) with K4L's age limit 2 -> 3 and the per-layer family 'k4la3' (K4L strength of
# the pairs born 3 layers earlier = K4L_C + EPS['k4la3'][l]). Shift k4la3 = -1 on every layer reproduces otr_m_v5 bit for bit. Nothing else changed.
# otr_m_v5 (trainer, 10-11 21:08Z): otr_m_v4.py + the early edge's per-layer scale as a per-layer family 'edx' (comment EARLY EDGE: scale = EDGEXS[l] +
# EPS['edx'][l]), so forward arms can move it through parameter files; workers import it as otr_m via worker_generated_v11.py. v4's lines follow.
# otr_m_v4 (trainer, 10-11 18:45Z): otr_m_v3.py + the STAR STRENGTH family 'ssb' (comment STAR STRENGTH); workers import it under the name otr_m via
# worker_generated_v10.py. Lines of v3 kept as they were (v3's header lines follow).
# otr_m_v3 (trainer, 10-11 17:35Z): otr_m_v2.py + the READ COLUMN CUT twin of the engineer's native S48 read cut (comment READ COLUMN CUT); workers import it
# under the name otr_m via worker_generated_v9.py. Lines of v2 kept as they were (v2's own header line follows).
# otr_m_v2 (trainer, 10-11 15:50Z): the trainer's VERSIONED card module = otr_m.py as of 15:30:24Z (sha fc2d4287; adds the default-off EDGE INPUTS and EARLY EDGE switches to bba0dea0). Workers import it under the name otr_m via worker_generated_v8.py; the live otr_m.py stays at bba0dea0 for everyone's pinned tools (lead 15:4xZ).
"""TAG opus-trainer (2026-10-02): the RUN M card = today's torch card + F4 + K4L with TRAINABLE strengths (lead 01:00 / 01:38 / 03:17 UTC).
F4 (physicist, O/physicist/f4.py, rule A, cheap anchor): at the end of step l (0..14) the carried columns born at most F4_AGE layers
  earlier and not yet joined are shifted in place, u_r -> u_r + theta A_r w_r, and the plane this adds to the bank is taken out of
  the star. Here theta is one number per column AGE (1 or 2 layers) and layer: theta = F4_THETA + EPS['f4a<age>'][l].
K4L (lead, O/K4L_SPEC.md, O/e16b_k4_closed_loop.py): the per-unit fourth cumulant gets the hub terms of the pairs born at most
  K4L_AGE layers earlier, their isotropic part projected out; pair map of layers 1-14 (argument and ctx['k4'], which the
  covariance-output hook reads) and the final layer. Here the strength is one number per pair age and layer:
  loc = sum over ages (K4L_C + EPS['k4la<age>'][l]) x (that age's hub terms), k4 += loc - beta(loc) v^2.
k22: scale of the OFF-DIAGONAL part of the isotropic (2,2) plane the pair map receives, layers 1-14: 1 + EPS['k22'][l].
e2: the second-order skew term the mean formula lacks (lead, arm e2 of e16b), layers 1-15, added through the correction slot:
  EPS['e2'][l] x k3^2 (alpha^4 - 6 alpha^2 + 3) pdf / (72 sigma^5); base 0. It is NOT scaled by ds.
F4C (env TRAINER_F4C=1): the F4 block of this module is switched off and the physicist's exact-term module takes its place
  (O/physicist/f4b.py, mode 'mix', metric 'birth', star anchor; it chains after this module's hooks): the rule's strength per age
  and layer is the same leaf as F4's theta (f4a1, f4a2; with the exact strengths at 0 the module is F4 to 1e-7 rms), and the
  strengths of the exact S1 and S2 terms per pair age are new leaves x1a1, x1a2, x2a1, x2a2 (base 0). Its birth data stay
  detached constants (the physicist's convention), so the gradients of the dials lack the paths through them; forward runs are exact.
NCAR (lead, O/ncar.py, O/NCAR_SPEC.md): a free-running vector carrier of the network-specific norm coupling of the fourth cumulant,
  used as an extra K31 plane through the pair map's remainder31 input. Installed LAST (after K4L and F4 / the exact-term module)
  when the module flag NCAR is a dict (the worker sets it from a request's 'ncar' field; {} = the arguments below as they are, any
  key overrides one). The install call's arguments are in ONE place, NCAR_ARGS; the three strengths are leaves read when the
  module uses them: nk31 (plane strength per layer, base 1), nct (transport, base 0.9), ncs (sources, base 1.0).
  {'module': 'ncar4', ...} installs the lead's SKETCHED rank-r carrier O/ncar4.py instead (arguments NCAR4_ARGS; only the plane
  strength nk31 is a leaf there; it also reads the push's lam argument and its old-tier column count from the side table).
  {'module': 'ncarry', ...} installs the physicist's FULL-MATRIX research carrier O/physicist/ncarry.py (arguments NCARRY_ARGS:
  complete certified law, conversion from the bank columns, plane minus its four forms, layers 1-14; any of its keywords may be
  given, e.g. state_rank, skip, ownrow, oneside, basis, ritz, law_ck); its plane strength is the leaf nk31 as well.
  {'module': 'ncar5', ...}: the lead's structured low-rank state without probes (O/ncar5.py, NCAR5_ARGS); {'module': 'bank', ...}:
  the physicist's bank channel, a second weight per bank column (O/physicist/ncar_bank_frozen_s2s.py, BANK_ARGS; its plane
  strength is the leaf nkbk). The request field may also be a LIST of such dicts: the modules are installed in that order and
  their planes add (the lead's 'both' = [bank, ncar5]).
  {'module': 'fold', ...}: the physicist's bank channel FOLDED INSIDE THE BANK READ (O/physicist/ncar_fold_frozen_s2w.py, FOLD_ARGS:
  level 'read', the multiple of tr C x C from the diagonal regression times dscale); {'module': 'ncar6', ...}: the lead's
  pair-carried lean law (O/ncar6.py, NCAR6_ARGS: level 'read', the multiple of C = per-layer literal x diagonal regression).
  These two wrap the card's bank read and chain the 'mu' and 'cpre' hooks themselves; nothing of this file sits on those.
  Their dials are leaves whose BASE is the request's (or the table's) value of the argument: k31 -> nfs (strength),
  dscale -> nfd (factor on the removed multiple of C), and for ncar6 births -> n6b, rows -> n6r, diag -> n6d (source strengths).
  ncar6 reads the layer's selected hubs and their two hub entries from the side table (side['births'][l]: sel, tp, tm).
  {'module': 'n6a', ...}: the physicist's audit copy of ncar6 (O/physicist/ncar6_audit_frozen_s3d.py): dfit 5 / 6 = the diagonal
  regression with a constant regressor (no per-layer table), k22 = strength of the K22 companion (leaf n6k), gain.
  A module's 'pair_in' may return a changed K21 plane, per-unit k4 or K22 plane besides its remainder31 plane: the change is added.
  The module's own
  work runs under no_grad (its state is detached anyway), so in a gradient run the plane is a constant and its leaves get no
  gradient: forward line searches only, as the lead asks. Since 15:30Z a gradient run with modules installed is VALID for the card's
  own arrays and dials: a layer step that the checkpoint runs again takes the modules' contributions from a record of the forward
  pass (comment REPLAY in install).
rso, rsy: scales on the old / young tier's column weights at the bank read only, layers with a read: 1 + EPS['rso'][l], 1 + EPS['rsy'][l] (base 1).
ds: scale of the learned mean correction, layers 1-15: delta x (1 + EPS['ds'][l]) (the lead's `ds` arm of e16b). Natively the
  correction of layer l is linear in the literals head[l] and head_b[l], so a scale is exported by multiplying those two (announce.py).
kc: kappa_C = 1 - EPS['kc'][l] (physicist T74, lead 10-10 05:32Z), layers 1-14: the share of the learned mean correction d that the step folds
  into the post covariance (C <- C - host d' - d host' - d d'); kc[l] x (host d' + d host' + d d') is given back after every other state_out
  hook, in the forward pass and when the step runs again (part of the graph). kc = 0 is the card bit for bit.
cvdd, cvhd, cvgd, cvsd, cvss, cvgg, cvmm (10-10 13:20Z, kappa_C's family): low-rank corrections of the post covariance handed on at layers 1-14, added
  after kappa_C: EPS[f][l] x (a b' + b a') (a b' once when a = b) with (a, b) = (d, d), (host, d), (g, d), (sigma, d), (sigma, sigma), (g, g), (m, m),
  d = the mean correction used, host = the pair map's mean, m = host + d, g = gate, sigma = the pre-activation sigma of the layer. Computed when any
  of them is nonzero or when CV is on (request field 'cv': 1, so that their exact gradients at 0 come back in 'pl_grads'); all 0 = the card bit for bit.
The leaves are added to otr_pl.EPS (families f4a1, f4a2, k4la1, k4la2, k22), so shifts travel in requests and parameter files and
their gradients come back in 'pl_grads' like every other per-layer dial. The base values F4_THETA = 0.6 and K4L_C = 1 are the
constants of the S37 file; shifts of -F4_THETA and -K4L_C on all layers give today's card back bit for bit, and equal shifts for
both ages reproduce the two reference modules bit for bit (m_certify.py).
Birth data (F4's scalar A, K4L's four hub coefficients and the hub entries of the newborn columns):
  BIRTH_ATTACH False (the two reference modules' convention): detached constants kept per layer in a side table, so recomputing
    a checkpointed step rewrites the same entries. The backward pass then sees the pushed columns move with the hub entries but
    not the hub entries themselves, although the forward value of K4L's two legs does not depend on them at all.
  BIRTH_ATTACH True (env TRAINER_BIRTH_ATTACH=1): the same numbers travel from step to step as a 15th element of the state
    tuple (a 'state_in' hook takes it off again before the card unpacks the state), so they stay inside the autograd graph
    across the per-layer checkpoints and the backward pass differentiates what the forward pass computes. Forward values equal.
(cert/otr_m_v1_detached_births.py is this file before the switch existed.)
usage (inside otr_engine when TRAINER_CARD=M): otr_m.install(card) right after otr_pl.install(engine, card)."""
import os
import sys

import torch

import otr_pl as PL

F4_THETA, F4_AGE, K4L_C, K4L_AGE = 0.6, 2, 1.0, 4   # scout-card sc3: K4L_AGE 2 -> 4
BIRTH_ATTACH = os.environ.get('TRAINER_BIRTH_ATTACH', '0') == '1'
F4C = os.environ.get('TRAINER_F4C', '0') == '1'
NEW = ('f4a1', 'f4a2', 'k4la1', 'k4la2', 'k4la3', 'k4la4', 'k22', 'ds', 'e2', 'x1a1', 'x1a2', 'x2a1', 'x2a2', 'nk31', 'nct', 'ncs', 'nkbk', 'ssb', 'edx',
       'nfs', 'nfd', 'n6b', 'n6r', 'n6d', 'n6k', 'nkl1', 'nkl2', 'nkl3', 'rso', 'rsy', 'kc',
       'cvdd', 'cvhd', 'cvgd', 'cvsd', 'cvss', 'cvgg', 'cvmm')
CV_FAMILIES = ('cvdd', 'cvhd', 'cvgd', 'cvsd', 'cvss', 'cvgg', 'cvmm')
CV = False
FEATS = False      # 10-10 20:55Z (lead 20:26Z): request field 'featstats': per layer 1-15 the raw head features' mean / std over the units and the share
FEATSTATS = {}     # clipped at +-8 by today's centers / scales, for a closed-form re-estimate of the September normalisation
NCAR = None                                                                    # None = no carrier; a dict = install it (keys override NCAR_ARGS)
NCAR_ARGS = dict(K=16, skip1=False, ca=0.5, cb=0.25, lo=2, hi=14, conv=1.0, law=2)   # the install call's arguments, in one place
NCAR4_ARGS = dict(r=32, k=24, K=8, conv=1, k21=True, lean=True, lo=2, hi=14, start=0, passes=2)   # the sketched carrier's (ncar4.py)
NCARRY_ARGS = dict(rank=0, k21=True, start=1, stop=14, conv='b', forms=4)     # the physicist's full-matrix carrier's (ncarry.py)
NCAR5_ARGS = dict(r=32, K=16, lo=1, hi=14, births=1.0, rows=1.0, k21=True, cleanup=True, ca=0.75, sweeps=0, extra=0, bankdiag=True)   # ncar5.py
BANK_ARGS = dict(lo=1, hi=14, forms=4, conv=1, columns=True, top=0)           # the physicist's bank channel (ncar_bank_frozen_s2s.py)
FOLD_ARGS = dict(k31=1.0, level='read', lo=1, hi=14, forms=1, dfit=3, dscale=1.41)   # the bank channel inside the bank read (ncar_fold_frozen_s2w.py)
NCAR6_ARGS = dict(k31=1.0, level='read', lo=1, hi=14, forms=1, dfit=2, dscale=1.0, ca=0.75, arest=False, births=1.0, rows=1.0, diag=1.0)   # ncar6.py
MODULES = dict(ncar=('ncar', NCAR_ARGS), ncar4=('ncar4', NCAR4_ARGS), ncar5=('ncar5', NCAR5_ARGS), ncarry=('ncarry', NCARRY_ARGS),
               bank=('ncar_bank_frozen_s2s', BANK_ARGS), fold=('ncar_fold_frozen_s2w', FOLD_ARGS),
               ncar6=('ncar6_t1339', NCAR6_ARGS))                             # request name -> (file, its install arguments)
# ncar6_t1339 = O/ncar6.py as of 13:39Z (frozen/ncar6_t1339.py, sha f87e683d; a folder that holds nothing else): a copy in this lane, so that a worker restarted in the
# middle of a dial solve loads the same law; with the options added at 13:39Z (cenb, cenr, ptop, pdiag, pmu) left at 0 it equals the 12:44Z file.
N6A_ARGS = dict(NCAR6_ARGS, gain=0, k22=0.0)   # the physicist's audit copy of ncar6 (dfit 5 / 6: constant regressor; k22: the K22 companion)
MODULES['n6a'] = ('ncar6_audit_frozen_s3d', N6A_ARGS)
LEAVES = dict(fold=dict(k31='nfs', dscale='nfd'),                             # module -> its arguments that are leaves (argument -> family);
              ncar6=dict(k31='nfs', dscale='nfd', births='n6b', rows='n6r', diag='n6d'))   # value = the argument's value + the leaf
LEAVES['n6a'] = dict(LEAVES['ncar6'], k22='n6k')                              # (k22 is a leaf only when the request asks for the K22 companion)
# n6c: O/ncar6.py as of 14:02Z (frozen/ncar6_t1402.py, sha 1897e1d4): dfit 3 = diagonal regression with a constant regressor (no table), dfit 4 = also C^2.
MODULES['n6c'] = ('ncar6_t1402', dict(NCAR6_ARGS, forms=3, dfit=3))
LEAVES['n6c'] = LEAVES['ncar6']
# k31l: the lead's fourth-chaos part of the young pairs' local quartic as a (3,1) plane in the same bank read (O/k31l.py as of 15:08Z, frozen/k31l_t1508.py,
# sha 39656462); it has no 'state_out' hook. Its strength is the leaf nkl2, or the family the request names under 'leaf' (several instances with different
# age limits and leaves nkl1 / nkl2 / nkl3 give one strength per cohort age).
MODULES['n6x'] = ('ncar6_t1402x', dict(NCAR6_ARGS, forms=3, dfit=3))            # = n6c with its state exposed (frozen/ncar6_t1402x.py): needed by the head inputs
LEAVES['n6x'] = LEAVES['ncar6']
K31L_ARGS = dict(k31=1.0, age=2, lo=1, hi=14, forms=0, chaos4=True, level='read')
MODULES['k31l'] = ('k31l_t1508', K31L_ARGS)
LEAVES['k31l'] = dict(k31='nkl2')
# n6y / k31ly: the same two modules with more of their state exposed (frozen/ncar6_t1402y.py, frozen/k31l_t1508y.py): the second batch of head inputs
MODULES['n6y'] = ('ncar6_t1402y', dict(NCAR6_ARGS, forms=3, dfit=3))
LEAVES['n6y'] = LEAVES['ncar6']
MODULES['k31ly'] = ('k31l_t1508y', K31L_ARGS)
LEAVES['k31ly'] = dict(k31='nkl2')
# pbase: the physicist's physics base (O/physicist/pbase_frozen_s4a.py; one keyword per term, all 0 = the card bit for bit), e.g. {'module': 'pbase', 'bsk': 1.0}
MODULES['pbase'] = ('pbase_frozen_s4a', dict(so2m=0.0, so2c=0.0, so2t=0.0, so2f=0.0, so2k=1, so2p=123456, so2td=0, bsk=0.0, gsk=0.0, gk4=0.0, gbk=0.0, x1=0.0, x2=0.0, lo=1, hi=14))
NCAR_BASE = dict(nk31=1.0, nct=0.9, ncs=1.0, nkbk=1.0)                        # strength = base + its leaf (per layer)
_NCAR = {}
# HEAD INPUTS (lead 17:52Z): new per-unit input columns of the learned mean correction, each with its own column of first-layer weights (128, the same
# for every layer) that starts at ZERO: the correction's first layer is silu(h @ enc0_w' + enc0_b) on 126 columns; a new column z enters as a per-unit
# bias, enc0_b + z_i u', so with u = 0 nothing changes bit for bit and the gradient of u at zero is exact. HEADF (per request, field 'hf') switches the
# six columns on (they need the law module 'n6x' first in the request and, for columns 4-5, k31l behind it); HEADU holds the weights (6 x 128; request
# field 'hu' sets them, default zero). Columns, each standardised over the layer's units of the network (response rows for 2-5; others 0) and clipped
# at +-8, layers 1-14:
#   0 the unit-specific part of the law matrix's diagonal, (B_ii - the diagonal regression's fit) / (tr C v_i)
#   1 row norm of the law's fold term in the read, standardised as the card's own plane feature (term_ij / (v_i s_j))    2 its row sum against rho_ij
#   3, 4 the same two numbers for the k31l term (everything the modules behind the law add to the read)
#   5 the unit's share in the removed forms, fit_i / B_ii
#   6 column 0 IN THE NATIVE CONVENTION (the native fold holds the matrix's diagonal on the read's RESPONSE ROWS only; lead / engineer 18:13Z, 18:42Z):
#     on the response rows of the layer's read, x_i = (B_ii - fit_i) / (tr C v_i) with fit = least squares of B_ii on (tr C v_i, 1, mu_i^2, 2 mu_i (C mu)_i)
#     OVER THOSE ROWS; z_i = clip((x_i - mean) / sd, -8, 8) with mean and sd (n - 1 in the denominator) over those rows; every other unit gets 0;
#     layers 1-14 (0 at layer 15, which has no read). float64 until the last step, then float32.
# SECOND BATCH (lead 18:42Z; they need the modules 'n6y' and 'k31ly'; standardised and clipped like column 6):
#   7, 8 the two parts of the matrix's diagonal on the response rows, each over tr C v_i: the carried-column part sum_r w_r pp_ir^2 and the pair cross
#        part sum_r x_r pp_ir pp_i,partner(r) (column 6 is their sum minus the fit)
#   9    the diagonal of the young pairs' fourth-chaos plane (the k31l instance with age limit 2) over v_i^2, response rows
#   10, 11 the law's per-unit sources of the layer BEFORE pushed through this layer's weights like the mean, over sigma_i, ALL units:
#        W_l' rv (row factor) and W_l' (d o Pv) (variance tilt)
# HEADF: 1 = all columns; 2 = everything except columns 1-4: no extra bank read is made (columns 1-4 stay 0).
HEADF = False
HEADU = None
HEAD_COLUMNS = ('diag_resid', 'law_rownorm', 'law_rowsum_rho', 'k31l_rownorm', 'k31l_rowsum_rho', 'forms_share', 'diag_resid_rows',
                'diag_carried_rows', 'diag_cross_rows', 'k31l_diag_rows', 'push_rv', 'push_dPv',
                'k3_young', 'k3_old', 'k3_star', 'k3_bform', 'k4_local',
                'f15_diag', 'f15_k31l_diag', 'f15_k4_local', 'k4_local_rows',
                'b6_m2', 'b6_k3d2', 'b6_gk', 'b6_gb', 'b6_xk4', 'b6_rg', 'b6_qold', 'b6_eold', 'b6_qb', 'b6_f15',
                'diag_resid_rows_late', 'k4_local_rows_late', 'band_1_9', 'band_10_14', 'band_15',
                'b7_fold', 'b7_kmu', 'b7_kg', 'b7_qk', 'b7_e34', 'b7_s1')
# Since 10-10 03:30Z (engineer 02:14Z / 03:0xZ, lead 02:25Z) the final-layer columns 17-19 are on the KEPT ROWS of the final layer's row cut,
# rows = {i : |mu_i| <= 3.5 sqrt(vv_i)} at layer 15 (the native file pushes the bank into layer 15 only there): 17 = residual of dB on (vv, 1, mu^2) fitted over
# those rows; 18 = the young pairs' plane diagonal with the chaos4 column means taken over those rows; 19 = K4L's raw sum projected with beta over those rows;
# each standardised over those rows (n - 1), clipped at 8, 0 elsewhere. (Until then they used the layer-14 alive mask, which indexes other units.)
# BATCH 6 (lead 02:13Z; the physicist's O/physicist/b6cols.py, request field 'b6': 1, installed after this module): 21-29 = m2, k3d2, gk, gb, xk4, rg, qold, eold, qb
# at layers 1-14 standardised over the read's response rows; 30 f15 at layer 15 only, over the kept rows (27 qold / 28 eold at layer 15 until 10-10 04:20Z).
B6 = False
# HSPLIT (10-10 09:15Z, request field 'hsplit' {row: [first layer, new row]}): banded weights for one column: at layers >= first its standardised value
# goes to the new row (31 = column 6 late, 32 = column 20 late) and row `row` is 0 there; the new row's weights start as a copy of the old row's, so
# the card is unchanged until they move apart.
HSPLIT = {}
# HBANDS (10-10 09:35Z, lead 09:07Z, request field 'hbands': 1): the head's first-layer bias by layer band: rows 33 / 34 / 35 = the constant 1 on every unit
# at layers 1-9 / 10-14 / 15 (not standardised), so their 128 weights are a band-specific shift of enc0_b; zero weights = the card unchanged.
HBANDS = False
# BATCH 7 (10-10 10:20Z, lead 10:02Z; the physicist's O/physicist/b7cols.py from the adjoint screen T64, request field 'b7': 1, installed after batch 6):
# rows 36-41 = fold, kmu, kg, qk, e34, s1 at layers 1-14, standardised over ALL units (the module's convention 'a'), clipped at 8.
B7 = False
L15_ROWCUT_Z = 3.5
# FINAL-LAYER COLUMNS (lead 10-10 02:11Z item 2), only at layer 15 (zero at layers 1-14), into the final layer's head through the same first-layer bias:
#   17 the law matrix's diagonal at layer 15: B_ii = sum_r pp_ir (w_r pp_ir + x_r pp_i,partner(r)) on the columns pushed into 15 with the law's weights after
#      layer 14 (the lead's ncar6 option k4f), residual of its regression on (v_i, 1, mu_i^2), over (sum v) v_i
#   18 the diagonal of the young pairs' fourth-chaos plane at layer 15 (k31l instance with age limit 2, its plane on the columns pushed into 15) over v_i^2
#   19 the K4L local quartic's per-unit value at layer 15 over v_i^2 (as the final layer's k4 receives it)
#   17-19 are standardised over the units ALIVE at layer 14 (the state's previous_alive at layer 15) and 0 for the others.
#   20 column 16 in the RESPONSE-ROWS convention (engineer 10-02 20:12Z): K4L's per-unit value over v^2 on the response rows of the layer's read,
#      standardised over them, 0 elsewhere, layers 1-14.
# Layer-15 values need the modules 'n6y' and 'k31ly' (third TRAINER line of frozen/k31l_t1508y.py: the plane on any layer's columns).
# THIRD BATCH (lead 19:02Z): the parts of the per-unit skew and kurtosis that the head sees only as one sum, ALL units, standardised and clipped as above:
#   12 k3 of the young bank columns, sum over the young r of lam_r pp_ir^3, over v sigma     13 the same for the old tier's columns
#   14 k3 of the star (diagonal of the star plane) over v sigma     15 k3 of the b-form, 3 v_i b_i over v sigma = 3 b_i / sigma_i
#   16 the K4L local quartic's per-unit value (as it is added to k4, isotropic part projected out) over v^2
ROWS = {}                                                                     # response rows of the read per layer, last run with head inputs
# READOUTS (lead 19:51Z / 19:53Z / 20:01Z, batch 4): three linear readouts with weights at ZERO, so the card is unchanged and the gradient of the weights is
# exact (request field 'rg': 1; weights 'rgw' = dict(row=2 x 52, b=16 x 64, rad=16 x 64), default zero). They need the law module 'n6y'.
#   row  (o4) a ROW GAIN OF THE BANK READ: the plane the read returns (card's bank plane with the modules' folds, response rows x live columns; the diagonal
#        is overwritten by the card afterwards) times (1 + r_i), layers 2-14, r_i = q_i . (ROW[0] + (l - 3) ROW[1]) (one weight vector and its slope in the
#        layer, as the physicist's law grows linearly with the layer). q_i, each standardised over the response rows (mean, sd with n - 1) and clipped at 8:
#        0 log sigma_i; 1 the K4L local quartic over v_i^2 (k4 / v^2 as the pair map receives it, up to the layer's radial constant); 2 the row sum of the
#        returned plane against the pre-activation mean over v_i; 3 the diagonal column of this layer's read (head column 6 before standardisation);
#        4-51 the 48 message columns of the previous layer's state as the covariance edge forms them (w' src1, (w^2)' src2, (w^3)' src3, clipped at 8).
#   b    (o2) the carried b: b_i <- b_i + sigma_i r_l (state_i . B[l]), layers 1-14, state = this layer's 64 numbers per unit, sigma_i = sqrt of the post
#        variance the layer hands on, r_l = the layer's rms of b / sigma.
#   rad  (o3) the radial handed on: radial <- radial (1 + mean_i(state_i) . RAD[l]), layers 1-14.
#   k4m  (o5, lead 10-10) the per-unit kurtosis BEFORE the pair map: k4_i <- k4_i + v_i^2 (msg_i . K4M[l]), layers 1-14 (also the covariance output's k4)
#   k3m  (o6) the per-unit skew on the K21 plane's diagonal before the pair map: k3_i <- k3_i + v_i sigma_i (msg_i . K3M[l]), layers 1-14
#        msg = the 48 message columns of the previous layer's state as the covariance edge forms them, standardised over the layer's units and clipped at 8.
RG = False
RG_W = {}
# DUMP15 (lead 10-10 03:05Z: the last layer's own correction fitted offline): request field 'dump15' = the path of an npz written at layer 15 with, per unit,
# H = the head's 126 inputs exactly as gnn_step forms them (30 features after centre / scale / clip and the layer-15 mask, then the 96 message columns),
# Z = the new head columns of layer 15 (all 21 columns; only 17-19 are not 0 there), S_in = the state handed on from layer 14 (64), S_out = the encoder's
# output at 15 (64), delta_head = gnn_step's delta, delta_final = what is added to the raw mean (delta_head x (1 + ds[15]) + the e2 term), host = the raw
# mean, final = host + delta_final, sigma; scalars rms15, gain15, ds15. Needs the head inputs on ('hf').
DUMP15 = None
RG_SHAPES = dict(row=(2, 52), b=(16, 64), rad=(16, 64), k4m=(16, 48), k3m=(16, 48))
# EDGE INPUTS (10-11, lead 'GO: edge new-input zero-weight gate'): new per-unit input columns of the learned covariance edge (layers 8-14), each with its
# own weights into the edge's 16 left and 16 right pre-activations, left_i += z_i . EDGEU[l, :, :16], right_j += z_j . EDGEU[l, :, 16:], starting at
# ZERO: with zero weights the edge adds an exact zero and the gradient of the weights is exact (per layer; shared weights' gradient = the sum over 8-14).
# Request field 'edgein': 1 (needs the head inputs 'hf' and batch 6 'b6'); 'eu' sets the weights (16 x 10 x 32, default zero). The edge function is the
# card's own translation of the source's _message_edge_source with '+ context[zl]' / '+ context[zr]' appended to its left / right lines.
# Columns z (n x 10), each standardised over the layer's units and clipped at 8 (0-3 exactly as the head sees them, z of comment HEAD INPUTS):
#   0 head column 6 (law diagonal residual, response rows)   1 head column 20 (K4L, response rows)   2 head column 27 (qold)   3 head column 28 (eold)
#   4 K21 row mean, sum_{j != i} S_ij / (n - 1), S_ij = K21_ij / (s_i^2 s_j) as the edge forms it   5 its column mean (unit j: sum_{i != j} S_ij / (n - 1))
#   6 K21 row rms   7 rho row mean, sum_{j != i} rho_ij / (n - 1), rho = the edge's pre_cov / (s s')   8 rho row rms
#   9 the previous layer's mean correction pushed into this layer, (W_l' d_{l-1})_i / s_i
EDGEIN = False
EDGEU = None
EDGE_COLUMNS = ('diag_resid_rows', 'k4_local_rows', 'b6_qold', 'b6_eold', 'k21_rowmean', 'k21_colmean', 'k21_rowrms', 'rho_rowmean', 'rho_rowrms',
                'dpush')
EDGE_HEAD_ROWS = (6, 20, 27, 28)
# EARLY EDGE (10-11 15:40Z, trainer idea after the RESTART): the learned covariance edge (trained for layers 8-14, frozen weights) ALSO applied at layers
# lo..hi outside 8-14, C <- C + EDGEXS[l] x edge source (the source called with layer 8 so its layer check passes; covhunter's edge_extend.py),
# chained after the covariance knobs' 'cov_pm' hook. EDGEXS = per-layer scales, a leaf (16,) starting at ZERO: the card unchanged bit for bit and the
# gradient of each layer's scale exact. Request fields 'edgex': [lo, hi], 'exs': 16 scales (default zero); worker_generated_v7.py records 'ex_grads'.
EDGEX = None
EDGEXS = None
# READ COLUMN CUT (10-11 17:35Z; the engineer's native rule for S48, engineer/a62_modules/make_rd_v1.py / make_rdz_v1.py): at the layers with a bank
# read (1-14) the read keeps only the prefix c = min(L, ceil32(#{units with g >= Phi(-RDC)})) of its live-roster columns (the roster = units with
# g >= 0.01 sorted by g descending, L of them, as the card's first select of the step returns it; the count is over all units): the read's whole
# output (young + old bank plane, the law's fold and forms, the k31l plane) is zeroed on the cut columns live[c:] on every row, BEFORE the card
# fills the plane's diagonal (still filled for every unit). Rows, the response / omitted sets and the radial source, births, pushes and the head's
# and edge's inputs are untouched. Request field 'rdc' (e.g. 2.0; None = off). c = L gives the card bit for bit.
RDC = None
# STAR STRENGTH (10-11 18:45Z, lead): with the one-direction star ('ss' -1, the engineer's star_ss_torch_v2 / native SS v3 form) a strength beta on the
# part beyond the row-sum form: s2 = s2_K0 + beta (s2_K-1 - s2_K0), i.e. the second factor T2 = T2_K0 + beta (T2_K-1 - T2_K0) with the common first
# factor P1 = W' (A 1); beta = 1 + EPS['ssb'][l] at the star's layer (1-15), forward only (a Python float: no gradient). Request field 'ssb': 1 (needs
# 'ss' -1). beta = 1 takes T2_K-1 as the star module computes it: the card bit for bit.
SSB = False


def readout_weights(values=None):
    """The leaves of the three readouts; set to the given values (dict by name) or to zero for the next run."""
    for name, shape in RG_SHAPES.items():
        if name not in RG_W:
            RG_W[name] = torch.zeros(shape, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            RG_W[name].zero_()
            if values and name in values:
                RG_W[name].copy_(torch.tensor(values[name], dtype=torch.float32).reshape(shape))
        RG_W[name].grad = None
    return RG_W


def head_weights(values=None):
    """The leaf of the new columns' first-layer weights; set to the given 6 x 128 values (or to zero) for the next run."""
    global HEADU
    if HEADU is None:
        HEADU = torch.zeros(len(HEAD_COLUMNS), 128, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        HEADU.zero_()
        if values is not None:      # rows given for the first columns; the others stay at zero
            given = torch.tensor(values, dtype=torch.float32)
            HEADU[:given.shape[0]].copy_(given)
    HEADU.grad = None
    return HEADU


def edgex_scales(values=None):
    """The leaf of the early edge's per-layer scales (16,); set to the given values (or to zero) for the next run."""
    global EDGEXS
    if EDGEXS is None:
        EDGEXS = torch.zeros(16, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        EDGEXS.zero_()
        if values is not None:
            EDGEXS.copy_(torch.tensor(values, dtype=torch.float64))
    EDGEXS.grad = None
    return EDGEXS


def edge_weights(values=None):
    """The leaf of the edge inputs' weights (16 layers x 10 columns x 32 = left | right); set to the given values (or to zero) for the next run."""
    global EDGEU
    if EDGEU is None:
        EDGEU = torch.zeros(16, len(EDGE_COLUMNS), 32, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        EDGEU.zero_()
        if values is not None:
            EDGEU.copy_(torch.tensor(values, dtype=torch.float32))
    EDGEU.grad = None
    return EDGEU


def carrier_arguments(request):
    """(module name, the install arguments) for a request's 'ncar' field."""
    name = request.get('module', 'ncar')
    return name, dict(MODULES[name][1], **{key: value for key, value in request.items() if key not in ('module', 'leaf')})


def _ncar(name='ncar'):
    """One of the lead's carrier modules, loaded on first use."""
    if name not in _NCAR:
        import importlib
        from otr_common import O
        for folder in (str(O), str(O / 'physicist'), str(O / 'trainer' / 'frozen')):
            if folder not in sys.path:
                sys.path.insert(0, folder)
        _NCAR[name] = importlib.import_module(MODULES[name][0])
    return _NCAR[name]


class _Leaf:
    """A strength that is read from its leaf at the moment a module multiplies with it."""
    def __init__(self, read):
        self.read = read

    def __mul__(self, other):
        return self.read() * other
    __rmul__ = __mul__

    def __bool__(self):
        return True
_F4B = []


def _f4b():
    """The physicist's module, loaded on first use (it has no imports of its own)."""
    if not _F4B:
        from otr_common import O
        sys.path.insert(0, str(O / 'physicist'))
        import f4b
        _F4B.append(f4b)
    return _F4B[0]


class _ByAge(dict):
    """cs / ths of f4b.install: a dict by pair age whose values are read when the module asks, on the layer the card is at."""
    def __init__(self, make):
        super().__init__({1: None, 2: None})
        self.make = make

    def __getitem__(self, age):
        return self.make(age)
PL.FAMILIES = PL.FAMILIES + NEW
PL.EPS.update({family: torch.zeros(16, dtype=torch.float64, requires_grad=True) for family in NEW})


KB = None         # scout-card sc4: b <- b - KB (m - host) (None / 0 = off)
BFIXG = None      # scout-card sc2: the gain b-fix gamma for this network (None = the card's knob as built, 0.1)


def install(cd):
    if BFIXG is not None:      # scout-card sc2
        cd.knobs['bfix_gamma'] = float(BFIXG)
    if KB:      # scout-card sc4: b <- b - KB (m - host) after the card's own b_post hook (bscale)
        _b_post_prev = cd.hooks.get('b_post')
        def _b_post_kb(l, b, ctx, _prev=_b_post_prev, _kb=float(KB)):
            b = _prev(l, b, ctx) if _prev is not None else b
            return b - _kb * (ctx['m'] - ctx['host']).to(b.dtype)
        cd.hooks['b_post'] = _b_post_kb
    t, eps = torch, PL.EPS
    side = dict(base={-1: t.zeros(0, dtype=t.float32)}, births={})
    push0, select0, bd0 = cd.shared['push'], cd.select, cd.env['_bd_update']

    def push(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
        r = push0(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=rank)
        side['pp'], side['oc'], side['starts'] = r[0], int(r[4]), (0,) + tuple(int(x) for x in stops)
        side['lam_in'], side['old'] = lam, int(r[4])                 # what the sketched carrier reads besides 'pp'
        side['stops'] = side['starts']                               # the lead's name for the cohort boundaries with a leading 0 (k31l)
        return r
    cd.shared['push'] = push

    # rso / rsy (lead 16:35Z, from the physicist's age profile of the third-order plane): one scale on the OLD tier's column weights and one on the
    # young tier's, AT THE BANK READ ONLY (the K21 plane on the response rows; the carried weights, the plane's diagonal and the other consumers
    # of lam are untouched): lam -> lam x (1 + EPS['rso'][l]) for the old columns, x (1 + EPS['rsy'][l]) for the young ones. The innermost wrapper
    # of the read, so the carrier modules' folds and the replay of gradient runs sit on top of it; zero shifts multiply by exactly 1.
    read_card = cd.shared['read']

    def read_scaled(tt, pp, lam, response_indices, indices, U, F, old_columns):
        l = cd.call_layer
        scale = t.cat(((1.0 + eps['rso'][l]).to(lam.dtype).expand(old_columns), (1.0 + eps['rsy'][l]).to(lam.dtype).expand(lam.numel() - old_columns)))
        return read_card(tt, pp, lam * scale, response_indices, indices, U, F, old_columns)
    cd.shared['read'] = read_scaled

    def select(score):   # the last selection of a step is the one of the births
        r = select0(score)
        side['sel'] = r[0]
        return r
    cd.select = select

    def bd(bp, Cpre, K21, aux, d, alive, births, a, plus, minus, coefficients):
        side['plus'], side['minus'] = plus, minus
        return bd0(bp, Cpre, K21, aux, d, alive, births, a, plus, minus, coefficients)
    cd.env['_bd_update'] = bd

    def state_in(l, state, ctx):   # the birth data of the last cohorts, when they travel with the state
        side['in'] = state[14] if len(state) == 15 else None
        side['hstate_in'] = state[8]                                 # the previous layer's state (row-gain readout: its message columns)
        side['alive_in'] = state[6]                                  # the previous layer's alive mask (final-layer head columns)
        return state[:14]
    assert 'state_in' not in cd.hooks
    cd.hooks['state_in'] = state_in
    known = lambda: side['in']['births'] if side['in'] is not None else side['births']

    # K4L: hub terms of the young pairs on the pushed columns of layer l, isotropic part projected out
    def local(l, vv, mu):
        return project(local_loc(l), vv, mu)

    def project(loc, vv, mu, rows=None):      # the isotropic part projected out (over the given rows only, for the native final-layer column 19)
        al = mu / vv.clamp_min(1e-12).sqrt()
        w4 = ((al * al - 1) * t.exp(-al * al / 2)) ** 2 + 1e-12
        iso = vv * vv
        if rows is None:
            beta = (w4 * loc * iso).sum() / (w4 * iso * iso).sum()
        else:
            beta = (w4 * loc * iso)[rows].sum() / (w4 * iso * iso)[rows].sum()
        return loc - beta * iso

    def local_loc(l):
        pp, starts = side['pp'].to(t.float64), side['starts']
        loc = t.zeros(1024, dtype=t.float64)
        for j in range(max(l - K4L_AGE, 0), l):
            B = known()[j]
            k0 = starts[j]
            K = (starts[j + 1] - k0) // 2
            assert K == len(B['tp']), (l, j, K, len(B['tp']))
            gap = (B['tp'] - B['tm'])[None, :]
            pplus, pminus = pp[:, k0:k0 + K], pp[:, k0 + K:k0 + 2 * K]
            Wt = (pplus - pminus) / gap
            Y = (B['tp'][None, :] * pminus - B['tm'][None, :] * pplus) / gap
            S1, S2, S3, S4 = B['S']
            part = (4 * S1[None, :] * Y ** 3 * Wt + 6 * S2[None, :] * Y ** 2 * Wt ** 2 + 4 * S3[None, :] * Y * Wt ** 3
                    + S4[None, :] * Wt ** 4).sum(1)
            loc = loc + (K4L_C + eps['k4la%d' % (l - j)][l]) * part
        return loc

    pair0 = cd.hooks['pair_in']
    def pair_in(l, pin, ctx):
        pin = pair0(l, pin, ctx)
        if l == 0:
            return pin
        dk = local(l, t.diagonal(ctx['Cpre']), pin[0])
        side['k4l_dk'] = dk                                          # the local quartic's per-unit value of this layer (head input 16)
        ctx['k4'] = ctx['k4'] + dk
        k22 = t.diagonal_scatter(pin[4] * (1.0 + eps['k22'][l]), t.diagonal(pin[4]))
        return (pin[0], pin[1], pin[2], pin[3] + dk, k22, pin[5])
    cd.hooks['pair_in'] = pair_in

    delta0 = cd.hooks['delta']
    def delta(l, value, ctx):
        value = delta0(l, value, ctx)
        if l == 0:
            return value
        aux = ctx['aux']
        a, k3 = aux['alpha'], aux['pre_k3']
        e2term = (k3 * k3 / 72) * (a ** 4 - 6 * a * a + 3) * aux['pdf'] / aux['sigma'] ** 5
        out = value * (1.0 + eps['ds'][l]) + eps['e2'][l] * e2term
        if l == 15 and DUMP15 and side.get('d15') is not None:      # the layer-15 dump (comment DUMP15)
            import numpy as np
            d = side.pop('d15'); host = ctx['host'].detach().to(t.float64); fin = out.detach().to(t.float64)
            arrays = {k: (v.detach().to(t.float32).numpy() if t.is_tensor(v) else np.float64(v)) for k, v in d.items()}
            np.savez_compressed(DUMP15, host=host.numpy(), delta_final=fin.numpy(), final=(host + fin).numpy(), ds15=np.float64(1.0 + float(eps['ds'][l])),
                                e2_term=(float(eps['e2'][l]) * e2term).detach().to(t.float64).numpy(), **arrays)
        return out
    cd.hooks['delta'] = delta

    k4_15 = cd.hooks['k4_15']

    def k4_final(l, k4, ctx):
        loc = local_loc(15)
        dk = project(loc, ctx['vv'], ctx['mu'])
        side['f15'] = dict(vv=ctx['vv'].detach(), mu=ctx['mu'].detach(), dk=dk.detach(), loc=loc.detach())          # the final-layer head columns
        return k4_15(l, k4, ctx) + dk
    cd.hooks['k4_15'] = k4_final

    out0 = cd.hooks['state_out']
    def state_out(l, out, ctx):
        out = out0(l, out, ctx)
        if l > 14:
            return out
        aux = ctx['aux']
        C, p, lam, star, alive, stops = out[1], out[2], out[3], out[4], out[6], out[12]
        # F4: shift the young carried columns, take the plane the shift adds out of the star (cheap anchor)
        A, pp, oc = (side['in']['A'] if side['in'] is not None else side['base'][l - 1]), side['pp'], side['oc']
        nc = int(pp.shape[1])
        assert A.numel() == nc, (l, A.numel(), nc)
        if l - F4_AGE >= 1:
            oc = max(oc, int(stops[l - F4_AGE - 1]))          # cohort j ends at column stops[j] and is l - j layers old
        if nc > oc and not F4C:
            young = max(oc, int(stops[l - 2])) if l >= 2 else oc   # first column of the cohort born one layer ago
            theta = t.cat(((F4_THETA + eps['f4a2'][l]).to(t.float32).expand(young - oc),
                           (F4_THETA + eps['f4a1'][l]).to(t.float32).expand(nc - young)))
            dh = ((aux['pdf'] / aux['sigma'] / 2) * alive).to(t.float32)
            u = p[:, oc:nc]
            om = dh[:, None] * (pp[:, oc:] * pp[:, oc:])
            new = u + om * (theta * A[oc:])[None, :]
            c = (lam[oc:nc] * theta * A[oc:]).to(t.float64)
            v = t.diagonal(C)
            u64, o64 = u.to(t.float64), om.to(t.float64)
            uu = u64 * u64
            vP = o64 @ (c * (v @ uu)) + 2 * (u64 @ (c * (v @ (u64 * o64))))
            beta = vP / (v * v).sum()
            dP = t.diagonal_scatter(v[:, None] * beta[None, :] + 2 * C * beta[:, None], 3 * (uu * o64) @ c)
            star = star - (3 * dP.T - 2 * t.diag(t.diagonal(dP))).to(star.dtype)
            p = t.cat((p[:, :oc], new, p[:, nc:]), dim=1)
        # birth data of this layer's pairs: F4's scalar A, K4L's hub entries and four coefficients
        sel = side['sel']
        keep = (lambda x: x) if BIRTH_ATTACH else (lambda x: x.detach())
        An = keep(-aux['alpha'][sel] / (aux['sigma'][sel] * (1 + ctx['radial'] / 3))).to(t.float32)
        f = lambda x: keep(x).to(t.float64)[sel]
        a, s, gt, ph, m = f(aux['alpha']), f(aux['sigma']), f(aux['gate']), f(aux['pdf']), f(ctx['host'])
        vv = s * s
        dj = ph / s
        r2 = vv * ((1 + a * a) * gt + a * ph)
        r3 = vv * s * ((a ** 3 + 3 * a) * gt + (a * a + 2) * ph)
        r4 = vv * vv * ((a ** 4 + 6 * a * a + 3) * gt + (a ** 3 + 5 * a) * ph)
        S = (-a * ph / vv, 2 * (gt - gt * gt - m * dj), 3 * (1 - gt) * (r2 - 2 * m * m),
             r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2)
        born = dict(tp=f(side['plus']), tm=f(side['minus']), S=S, sel=sel)
        out = out[:2] + (p, lam, star) + out[5:]
        if BIRTH_ATTACH:
            side['births'][l] = dict(tp=born['tp'].detach(), tm=born['tm'].detach(), sel=sel, S=tuple(x.detach() for x in S))   # what a carrier module reads (ncar6, k31l)
            return out + (dict(A=t.cat((A, An, An)), births={**{j: B for j, B in known().items() if j > l - K4L_AGE}, l: born}),)
        side['base'][l], side['births'][l] = t.cat((A, An, An)), born
        return out
    cd.hooks['state_out'] = state_out
    cd.m_side = side
    if F4C:   # the physicist's exact-term module in place of the F4 block (it wraps push, select, the birth update and 'state_out')
        layer = lambda: cd.call_layer
        _f4b().install(cd, t, 1.0, age=F4_AGE, mode='mix', metric='birth',
                    cs=_ByAge(lambda age: (eps['x1a%d' % age][layer()], eps['x2a%d' % age][layer()], 0.0, 0.0)),
                    ths=_ByAge(lambda age: (F4_THETA + eps['f4a%d' % age][layer()],) * 4))
    if NCAR is not None:   # carrier modules, last: each adds a remainder31 plane in 'pair_in' and keeps its own state in 'state_out'
        mine_in, mine_out = cd.hooks['pair_in'], cd.hooks['state_out']
        read_plain, before = cd.shared['read'], {name: cd.hooks.get(name) for name in ('mu', 'cpre')}
        read_law = None
        strength = lambda family: _Leaf(lambda: NCAR_BASE[family] + eps[family][cd.call_layer])
        parts = []
        for request in (NCAR if isinstance(NCAR, list) else [NCAR]):
            cd.hooks['pair_in'], cd.hooks['state_out'] = None, None      # a module chains what it finds; here it must find nothing
            name, arguments = carrier_arguments(request)
            leaf = strength('nkbk' if name == 'bank' else 'nk31')
            if name == 'ncar':
                arguments = dict(arguments, ct=strength('nct'), cs=strength('ncs'))
            if name in LEAVES:        # dials as leaves around the request's values
                families = dict(LEAVES[name], **({'k31': request['leaf']} if 'leaf' in request else {}))
                dials = {key: _Leaf(lambda base=arguments[key], family=family: base + eps[family][cd.call_layer])
                         for key, family in families.items() if key != 'k22' or arguments[key]}
                _ncar(name).install(cd, t, side, **dict(arguments, **dials))
            elif name == 'ncarry':    # no side table (it wraps the push and 'state_in' itself); tuples where its code unpacks or tests membership
                arguments = {key: tuple(value) if isinstance(value, list) else value for key, value in arguments.items()}
                _ncar(name).install(cd, t, strength=leaf, **arguments)
            else:
                _ncar(name).install(cd, t, side, k31=leaf, **arguments)
            parts.append((cd.hooks['pair_in'], cd.hooks['state_out']))
            if name in ('n6x', 'n6y'):
                read_law = cd.shared['read']                                # the read with the law's fold only (head inputs)

        # REPLAY (gradient runs). The per-layer checkpoints run a step once in the forward pass and AGAIN inside the backward pass (non-reentrant
        # checkpoint: autograd is on both times). The modules keep their state outside the graph and in layer order, so a step that runs again does
        # not call them: it takes what they added in the forward pass from a record (the fold term of the bank read, the remainder31 plane, the
        # changes of the K21 / k4 / K22 inputs). To the backward pass the carrier is therefore a CONSTANT of each layer: the gradients of the
        # card's arrays and dials lack the paths through the carrier's own dependence on the state, and the carrier's leaves get none.
        # Forward-only runs (no autograd) go through the modules exactly as before.
        record, seen = {}, set()
        step0 = cd.step

        def step(layer, state):
            side['again'], side['reads'], side['layer'] = layer in seen, 0, layer
            seen.add(layer)
            return step0(layer, state)
        cd.step = step
        read_modules = cd.shared['read']
        if read_modules is not read_plain:
            def head_raw(args, plain, full):      # per response row: norm and rho-weighted sum of the law's and of the later modules' fold terms
                tt, pp, lam, response_indices, indices, U, F, old_columns = args
                if HEADF == 2:      # diagonal columns only: just the read's response rows
                    side['hf_raw'] = dict(l=side['layer'], rows=response_indices, values=None)
                    return
                S6 = cd.ncar6_state
                law = read_law(*args)
                C = S6['C']; v = t.diagonal(C).clamp_min(1e-12); sd = v.sqrt()
                scale = (1.0 / v[response_indices])[:, None] / sd[indices][None, :]
                rho = C[response_indices][:, indices] / (sd[response_indices][:, None] * sd[indices][None, :])
                out = []
                for term in (law - plain, full - law):
                    term = term.to(t.float64) * scale
                    out += [(term * term).sum(1).sqrt(), (term * rho).sum(1)]
                side['hf_raw'] = dict(l=side['layer'], rows=response_indices, values=out)

            def messages_of(l):      # the 48 message columns of the previous layer's state, as the covariance edge forms them (n x 48, float64)
                w = cd.w[l]; w2 = w * w
                src = side['hstate_in'].detach() @ cd.P['msg_w'].detach().T + cd.P['msg_b'].detach()
                return t.cat((w.T @ src[:, :16], w2.T @ src[:, 16:32], (w2 * w).T @ src[:, 32:]), 1).clamp(-8, 8).to(t.float64)
            side['messages'] = messages_of

            def row_inputs(l, args, full):      # the 52 inputs of the row-gain readout on the response rows, standardised over them
                tt, pp, lam, rows, indices, U, F, old_columns = args
                S6 = cd.ncar6_state; C = S6['C']; mu = S6['mu']; v = t.diagonal(C).clamp_min(1e-12); trC = t.trace(C)
                x = t.zeros(rows.numel(), dtype=t.float64)
                if S6.get('diagB') is not None and l in (S6.get('fit_l'), S6.get('dparts_l')):
                    X = t.stack([trC * v, t.ones_like(v), mu * mu, 2 * mu * (C @ mu)], 1)[rows]; y = S6['diagB'][rows]
                    x = (y - X @ t.linalg.lstsq(X, y[:, None]).solution[:, 0]) / (trC * v[rows])
                first = t.stack([v.sqrt().log()[rows], (local(l, v, mu).detach() / (v * v))[rows], (full.to(t.float64) @ mu[indices]) / v[rows], x], 1)
                Q = t.cat((first, messages_of(l)[rows]), 1)
                spread = Q.std(0)
                Q = t.where(spread > 0, (Q - Q.mean(0)) / spread.clamp_min(1e-300), t.zeros_like(Q)).clamp(-8, 8)
                return Q.to(t.float32)

            def row_gain(l, q):
                return (1 + q @ (RG_W['row'][0] + (l - 3) * RG_W['row'][1]))[:, None]

            def read(tt, pp, lam, response_indices, indices, U, F, old_columns):
                args = (tt, pp, lam, response_indices, indices, U, F, old_columns)
                l = side['layer']
                if not t.is_grad_enabled():
                    full = read_modules(*args)
                    if HEADF:
                        head_raw(args, read_plain(*args) if HEADF == 1 else None, full)
                    if RG and 2 <= l <= 14:
                        full = full * row_gain(l, row_inputs(l, args, full))
                    return full
                key = ('read', l, side['reads'])
                side['reads'] += 1
                plain = read_plain(*args)
                if not side['again']:
                    with t.no_grad():
                        record[key] = read_modules(*args)
                        if HEADF:
                            head_raw(args, plain.detach(), record[key])
                        if RG and 2 <= l <= 14:
                            record[('rgq',) + key] = row_inputs(l, args, record[key])
                out = record[key] + (plain - plain.detach())      # the VALUE is the modules' read bit for bit (as a forward-only run), the gradient the card's own read's
                if RG and 2 <= l <= 14:
                    out = out * row_gain(l, record[('rgq',) + key])
                return out
            cd.shared['read'] = read
        if HEADF:
            assert read_law is not None and HEADU is not None, "head inputs need the module 'n6x' in the request and head_weights() called"
            gnn0 = cd.env['gnn_step']

            def columns(l):      # the six new input columns of layer l (n x 6, float32), standardised over the units they are defined on
                S6, raw, n = cd.ncar6_state, side.get('hf_raw'), 1024
                z = t.zeros((n, len(HEAD_COLUMNS)), dtype=t.float64)
                def put(k, x, mask=None):
                    x = x.to(t.float64)
                    y = x if mask is None else x[mask]
                    spread = y.std()
                    if bool(t.isfinite(spread)) and float(spread) > 0:
                        y = ((y - y.mean()) / spread).clamp(-8, 8)
                        if mask is None:
                            z[:, k] = y
                        else:
                            z[mask, k] = y
                if S6.get('fit_l') == l and S6.get('diagB') is not None:
                    C = S6['C']; v = t.diagonal(C).clamp_min(1e-12); B, fit = S6['diagB'], S6['fit']
                    put(0, (B - fit) / (t.trace(C) * v))
                    floor = 1e-3 * float(B.abs().median())
                    put(5, fit / t.where(B.abs() < floor, t.where(B < 0, -floor, floor) * t.ones_like(B), B))
                if raw is not None and raw['l'] == l:
                    rows = raw['rows']; ROWS[l] = int(rows.numel())
                    mask = t.zeros(n, dtype=t.bool); mask[rows] = True
                    for k, x in zip((1, 2, 3, 4), raw['values'] or ()):
                        full = t.zeros(n, dtype=t.float64); full[rows] = x
                        put(k, full, mask)
                    if S6.get('diagB') is not None and l in (S6.get('fit_l'), S6.get('dparts_l')):      # column 6: the native convention, response rows only
                        C = S6['C']; v = t.diagonal(C).clamp_min(1e-12); mu = S6['mu']; trC = t.trace(C)
                        X = t.stack([trC * v, t.ones_like(v), mu * mu, 2 * mu * (C @ mu)], 1)[rows]; y = S6['diagB'][rows]
                        resid = y - X @ t.linalg.lstsq(X, y[:, None]).solution[:, 0]
                        full = t.zeros(n, dtype=t.float64); full[rows] = resid / (trC * v[rows])
                        put(6, full, mask)
                    if S6.get('dparts_l') == l:      # columns 7, 8: the two parts of the diagonal
                        C = S6['C']; v = t.diagonal(C).clamp_min(1e-12); trC = t.trace(C)
                        for k, part in zip((7, 8), S6['dparts']):
                            full = t.zeros(n, dtype=t.float64); full[rows] = part[rows] / (trC * v[rows])
                            put(k, full, mask)
                    for age, Sk in getattr(cd, 'k31l_states', []):
                        if age == 2 and Sk.get('pdiag_l') == l:      # column 9: diagonal of the young pairs' fourth-chaos plane
                            v = t.diagonal(S6['C']).clamp_min(1e-12)
                            full = t.zeros(n, dtype=t.float64); full[rows] = Sk['pdiag'][rows] / (v[rows] * v[rows])
                            put(9, full, mask)
                if l <= 14 and S6.get('src_l') == l - 1 and S6.get('rv_prev') is not None:      # columns 10, 11: the previous layer's sources pushed through W
                    sd = t.diagonal(S6['C']).clamp_min(1e-12).sqrt(); w64 = cd.ws[l]
                    put(10, (w64.T @ S6['rv_prev']) / sd)
                    put(11, (w64.T @ S6['dPv_prev']) / sd)
                parts = side.get('k3parts')
                if parts is not None and parts['l'] == l:      # columns 12-16: the parts of the per-unit skew and kurtosis
                    for k, x in zip((12, 13, 14, 15, 16), parts['values']):
                        put(k, x)
                    if raw is not None and raw['l'] == l:      # column 20: column 16 on the response rows only
                        full = t.zeros(n, dtype=t.float64); full[raw['rows']] = parts['values'][4][raw['rows']]
                        put(20, full, mask)
                if l == 15 and side.get('f15') is not None:      # columns 17-19: the final layer, on the KEPT ROWS of its row cut
                    F15 = side['f15']; vv = F15['vv'].to(t.float64).clamp_min(1e-12); mu15 = F15['mu'].to(t.float64)
                    kept = mu15.abs() <= L15_ROWCUT_Z * vv.sqrt()
                    pp = side['pp'].detach().to(t.float64); ncol = pp.shape[1]
                    if S6.get('w') is not None and ncol:
                        w, x, partner = S6['w'][:ncol], S6['x'][:ncol], S6['partner'][:ncol]
                        dB = (pp * (pp * w[None, :] + pp[:, partner] * x[None, :])).sum(1)
                        X = t.stack([vv, t.ones_like(vv), mu15 * mu15], 1)[kept]
                        res = t.zeros(n, dtype=t.float64)
                        res[kept] = dB[kept] - X @ t.linalg.lstsq(X, dB[kept][:, None]).solution[:, 0]
                        put(17, res / (vv.sum() * vv), kept)
                    if ncol:      # column 18: the young pairs' fourth-chaos plane diagonal (ages 1-2, as k31l), chaos4 means over the kept rows
                        diag = t.zeros(n, dtype=t.float64); starts = side['starts']
                        for j in (13, 14):
                            B = side['births'].get(j); k0 = starts[j]; K = (starts[j + 1] - k0) // 2
                            if B is None or not K:
                                continue
                            tp, tm = B['tp'].to(t.float64)[None, :], B['tm'].to(t.float64)[None, :]; gap = tp - tm
                            pplus, pminus = pp[:, k0:k0 + K], pp[:, k0 + K:k0 + 2 * K]
                            E = (pplus - pminus) / gap; Y = (tp * pminus - tm * pplus) / gap; S1, S2, S3, S4 = (q.to(t.float64)[None, :] for q in B['S'])
                            AY = 3 * S1 * Y * Y * E + 3 * S2 * Y * E * E + S3 * E ** 3
                            AE = S1 * Y ** 3 + 3 * S2 * Y * Y * E + 3 * S3 * Y * E * E + S4 * E ** 3
                            mY2, mYE, mE2 = (Y * Y)[kept].mean(0)[None, :], (Y * E)[kept].mean(0)[None, :], (E * E)[kept].mean(0)[None, :]
                            AY = AY - 3 * (S1 * (mY2 * E + 2 * mYE * Y) + S2 * (mE2 * Y + 2 * mYE * E) + S3 * mE2 * E)
                            AE = AE - 3 * (S1 * mY2 * Y + S2 * (mY2 * E + 2 * mYE * Y) + S3 * (mE2 * Y + 2 * mYE * E) + S4 * mE2 * E)
                            diag = diag + (AY * Y + AE * E).sum(1)
                        put(18, diag / (vv * vv), kept)
                    put(19, project(F15['loc'].to(t.float64), vv, mu15, kept) / (vv * vv), kept)
                    b6 = getattr(cd, 'b6cols', {}).get(15)
                    if B6 and b6 is not None:      # column 30 at layer 15 (batch 6; qold / eold stay at layers 1-14 as the native columns, lead 10-10)
                        for k, name in ((30, 'f15'),):
                            if name in b6:
                                put(k, b6[name], kept)
                if B6 and 1 <= l <= 14 and getattr(cd, 'b6cols', {}).get(l) is not None:      # columns 21-29 (batch 6) on the response rows
                    b6 = cd.b6cols[l]; rmask = b6['resp'] > 0.5
                    for k, name in zip(range(21, 30), ('m2', 'k3d2', 'gk', 'gb', 'xk4', 'rg', 'qold', 'eold', 'qb')):
                        if name in b6:
                            x6 = b6[name].to(t.float64); full = t.zeros(n, dtype=t.float64); full[rmask] = x6[rmask]
                            put(k, full, rmask)
                if B7 and 1 <= l <= 14 and getattr(cd, 'b7cols', {}).get(l) is not None:      # columns 36-41 (batch 7) over all units
                    for k, name in zip(range(36, 42), ('fold', 'kmu', 'kg', 'qk', 'e34', 's1')):
                        if name in cd.b7cols[l]:
                            put(k, cd.b7cols[l][name])
                if HBANDS:      # band biases (comment HBANDS)
                    z[:, 33 if l <= 9 else 34 if l <= 14 else 35] = 1.0
                for row, (first, new) in HSPLIT.items():      # banded weights (comment HSPLIT)
                    if l >= first:
                        z[:, new] = z[:, row]
                        z[:, row] = 0.0
                return z.to(t.float32)

            def gnn_step(l, x, sigma, W, state, P):
                if FEATS and not side['again']:      # comment FEATS
                    with t.no_grad():
                        x64 = x.detach().to(t.float64); xs = (x.detach().to(t.float32) - P['centers'][l]) / P['scales'][l]
                        FEATSTATS[l] = dict(mean=x64.mean(0).tolist(), std=x64.std(0).tolist(), clip=(xs.abs() > 8).double().mean(0).tolist())
                if 1 <= l <= 15:
                    if side['again']:
                        z = record[('hf', l)]
                    else:
                        with t.no_grad():
                            z = columns(l)
                        if t.is_grad_enabled():
                            record[('hf', l)] = z
                    P = dict(P, enc0_b=P['enc0_b'][None, :] + z @ HEADU)
                if l == 15 and DUMP15:      # the head's inputs at layer 15, formed exactly as gnn_step forms them
                    with t.no_grad():
                        xs = ((x.to(t.float32) - P['centers'][l]) / P['scales'][l]).clamp(-8, 8)
                        xs = xs * t.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=t.float32)[None, :]
                        g = xs[:, 5:6]; src = state @ P['msg_w'].T + P['msg_b']; cm = int(P['msg_w'].shape[0]) // 3
                        w = W.to(t.float32); w2 = w * w; w3 = w2 * w
                        m1 = w.T @ src[:, :cm]; m2 = w2.T @ src[:, cm:2 * cm]; m3 = w3.T @ src[:, 2 * cm:]
                        e1 = w2.T @ (src[:, :cm] ** 2); e2_ = (w2 * w2).T @ (src[:, cm:2 * cm] ** 2); e3 = (w3 * w3).T @ (src[:, 2 * cm:] ** 2)
                        msgs = t.cat([g * m1, g * m2, g * m3, t.log1p(e1.clamp_min(0)), t.log1p(e2_.clamp_min(0)), t.log1p(e3.clamp_min(0))], 1).clamp(-8, 8)
                        side['d15'] = dict(H=t.cat([xs, msgs], 1).clone(), Z=z.clone(), S_in=state.detach().clone(), sigma=sigma.detach().clone(),
                                           rms15=float(P['rms'][l]), gain15=float(P['gain'][l]))
                out = gnn0(l, x, sigma, W, state, P)
                if l == 15 and DUMP15:
                    side['d15'].update(S_out=out[1].detach().clone(), delta_head=out[0].detach().clone())
                return out
            cd.env['gnn_step'] = gnn_step
            if EDGEIN:      # comment EDGE INPUTS
                import ast, copy
                import model_bank_research as MB
                assert B6 and EDGEU is not None, "edge inputs need batch 6 ('b6') and edge_weights() called"
                edge0 = cd.env['_message_edge_source']
                assert edge0.__code__.co_filename.endswith(':explicit_offline_translation')
                fn = copy.deepcopy(next(x for x in MB.extracted().body if x.name == '_message_edge_source'))
                fn.name, added = '_message_edge_source_ein', []
                for node in ast.walk(fn):
                    if isinstance(node, ast.Assign) and [getattr(x, 'id', None) for x in node.targets] in (['left'], ['right']):
                        key = 'zl' if node.targets[0].id == 'left' else 'zr'
                        node.value = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Subscript(
                            value=ast.Name(id='context', ctx=ast.Load()), slice=ast.Constant(value=key), ctx=ast.Load()))
                        added.append(key)
                assert sorted(added) == ['zl', 'zr']
                tree = ast.fix_missing_locations(MB.Translate().visit(ast.Module(body=[fn], type_ignores=[])))
                exec(compile(tree, edge0.__code__.co_filename + ':edge_inputs', 'exec'), cd.env)
                edge1 = cd.env.pop('_message_edge_source_ein')      # its globals stay the card's environment

                def standard(x):
                    spread = x.std()
                    if bool(t.isfinite(spread)) and float(spread) > 0:
                        return ((x - x.mean()) / spread).clamp(-8, 8)
                    return t.zeros_like(x)

                def edge_columns(l, K21, aux):
                    n = 1024
                    s = aux['sigma'].detach().to(t.float64)
                    off = 1 - t.eye(n, dtype=t.float64)
                    R = aux['pre_cov'].detach().to(t.float64) / t.outer(s, s) * off
                    S = K21.detach().to(t.float64) / (s[:, None] ** 2 * s[None, :]) * off
                    d = side.get('ein_delta')
                    push = (cd.ws[l].T @ d[1]) / s if d is not None and d[0] == l - 1 else t.zeros(n, dtype=t.float64)
                    stats = (S.sum(1) / (n - 1), S.sum(0) / (n - 1), ((S * S).sum(1) / (n - 1)).sqrt(), R.sum(1) / (n - 1),
                             ((R * R).sum(1) / (n - 1)).sqrt(), push)
                    return t.cat((columns(l)[:, list(EDGE_HEAD_ROWS)], t.stack([standard(x) for x in stats], 1).to(t.float32)), 1)

                def edge(layer, m, K21, aux, w, state, P, context):
                    if side['again']:
                        Z = record[('ein', layer)]
                    else:
                        with t.no_grad():
                            Z = edge_columns(layer, K21, aux)
                        if t.is_grad_enabled():
                            record[('ein', layer)] = Z
                    U = EDGEU[layer]
                    return edge1(layer, m, K21, aux, w, state, P, dict(context, zl=Z @ U[:, :16], zr=Z @ U[:, 16:]))
                cd.env['_message_edge_source'] = edge

                delta_in = cd.hooks['delta']
                def delta_kept(l, value, ctx):      # the layer's mean correction, for the next layer's column 9
                    out = delta_in(l, value, ctx)
                    if not side['again']:
                        side['ein_delta'] = (l, out.detach().to(t.float64).reshape(-1))
                    return out
                cd.hooks['delta'] = delta_kept
        for name in ('mu', 'cpre'):      # the modules' recording hooks are not run again either
            if cd.hooks.get(name) is not before[name]:
                cd.hooks[name] = (lambda l, value, ctx, modules=cd.hooks[name], plain=before[name]:
                                  (value if plain is None else plain(l, value, ctx)) if side['again'] else modules(l, value, ctx))

        messages = lambda l: side['messages'](l)

        def pair_in_with_carrier(l, pin, ctx):
            pin = mine_in(l, pin, ctx)
            if HEADF and not side['again'] and 1 <= l <= 14:      # the parts of the per-unit skew and kurtosis of this layer (head inputs 12-16)
                with t.no_grad():
                    v = t.diagonal(ctx['Cpre']).detach().clamp_min(1e-12); sd = v.sqrt()
                    p64, lam64, old = side['pp'].detach().to(t.float64), side['lam_in'].detach().to(t.float64)[:side['pp'].shape[1]], side['old']
                    cube = p64 * p64 * p64
                    side['k3parts'] = dict(l=l, values=((cube[:, old:] @ lam64[old:]) / (v * sd), (cube[:, :old] @ lam64[:old]) / (v * sd),
                                                        t.diagonal(ctx['sp']).detach() / (v * sd), 3 * ctx['bpre'].detach() / sd,
                                                        side['k4l_dk'].detach() / (v * v)))
            if side['again']:
                plane, changes, check = record[('pin', l)]
                now = float(pin[2].detach().double().abs().sum())
                assert abs(now - check) <= 1e-6 * check, ('the replayed bank read differs from the forward pass', l, now, check)
            else:
                plane, changes = None, {}
                with t.no_grad():
                    given = tuple(x.detach() if t.is_tensor(x) else x for x in pin[:5]) + (None,)
                    for carrier_in, _ in parts:
                        got = carrier_in(l, given, ctx)
                        extra = got[5]
                        if extra is not None:
                            plane = extra if plane is None else plane + extra
                        for i in (2, 3, 4):      # a module may also change the K21 plane, the per-unit k4 or the K22 plane it was given: its change is added
                            if got[i] is not given[i]:
                                changes[i] = changes.get(i, 0) + (got[i] - given[i])
                if t.is_grad_enabled():
                    record[('pin', l)] = (plane, changes, float(pin[2].detach().double().abs().sum()))
            if changes:
                pin = pin[:2] + tuple(pin[i] + changes[i].to(pin[i].dtype) if i in changes else pin[i] for i in (2, 3, 4)) + pin[5:]
            if RG and 1 <= l <= 14:      # readouts o5 / o6: per-unit kurtosis and skew before the pair map from the previous state's messages
                with t.no_grad():
                    msg = messages(l)
                    msg = ((msg - msg.mean(0)) / msg.std(0).clamp_min(1e-300)).clamp(-8, 8).to(t.float32)
                v = t.diagonal(pin[1]).detach().clamp_min(1e-12)
                dk4 = v * v * (msg @ RG_W['k4m'][l]).to(t.float64)
                dk3 = v * v.sqrt() * (msg @ RG_W['k3m'][l]).to(t.float64)
                ctx['k4'] = ctx['k4'] + dk4
                pin = pin[:2] + (pin[2] + t.diag(dk3).to(pin[2].dtype), pin[3] + dk4.to(pin[3].dtype)) + pin[4:]
            if plane is None:
                return pin
            return pin[:5] + (plane if pin[5] is None else pin[5] + plane,)
        cd.hooks['pair_in'] = pair_in_with_carrier

        def state_out_with_carrier(l, out, ctx):
            out = mine_out(l, out, ctx)
            if not side['again']:
                with t.no_grad():
                    for _, carrier_out in parts:
                        if carrier_out is not None:
                            carrier_out(l, out, ctx)
            if RG and 1 <= l <= 14:      # readouts of this layer's state: the carried b and the radial handed on (part of the graph: also when the step runs again)
                state = out[8].to(t.float64)
                sigma = t.diagonal(out[1]).detach().clamp_min(1e-12).sqrt()
                b = out[5]
                b = b + sigma * (b.detach() / sigma).pow(2).mean().sqrt() * (state @ RG_W['b'][l].to(t.float64))
                radial = out[7] * (1 + state.mean(0) @ RG_W['rad'][l].to(t.float64))
                out = out[:5] + (b,) + out[6:7] + (radial,) + out[8:]
            if 1 <= l <= 14 and bool(eps['kc'][1:15].abs().sum() > 0):      # kappa_C (comment kc)
                host, d = ctx['host'].to(out[1].dtype), ctx['delta'].to(out[1].dtype).reshape(-1)
                out = (out[0], out[1] + eps['kc'][l].to(out[1].dtype) * (t.outer(host, d) + t.outer(d, host) + t.outer(d, d))) + tuple(out[2:])
            if 1 <= l <= 14 and (CV or any(bool(eps[f][1:15].abs().sum() > 0) for f in CV_FAMILIES)):      # low-rank covariance corrections (comment cvdd ...)
                dt = out[1].dtype
                host, d = ctx['host'].to(dt), ctx['delta'].to(dt).reshape(-1)
                g, sg = ctx['aux']['gate'].to(dt), ctx['aux']['sigma'].to(dt); m = host + d
                add = 0
                for f, (a, b) in zip(CV_FAMILIES, ((d, d), (host, d), (g, d), (sg, d), (sg, sg), (g, g), (m, m))):
                    add = add + eps[f][l].to(dt) * (t.outer(a, a) if a is b else t.outer(a, b) + t.outer(b, a))
                out = (out[0], out[1] + add) + tuple(out[2:])
            return out
        cd.hooks['state_out'] = state_out_with_carrier
        if B6:      # the physicist's batch-6 columns (read-only hooks, installed after everything else)
            from otr_common import O
            if str(O / 'physicist') not in sys.path:
                sys.path.insert(0, str(O / 'physicist'))
            import b6cols
            b6cols.install(cd, t)
        if B7:      # the physicist's batch-7 columns (read-only hooks, after batch 6)
            from otr_common import O
            if str(O / 'physicist') not in sys.path:
                sys.path.insert(0, str(O / 'physicist'))
            import b7cols
            b7cols.install(cd, t)
    if SSB:      # comment STAR STRENGTH
        def ss_factors(wt, A32):
            rows = A32.sum(1)
            P1, T2 = (wt @ rows)[:, None], (((wt * wt) @ (rows @ A32)) / (rows @ rows))[:, None]      # star_ss_torch_v2, K < 0
            beta = 1.0 + float(eps['ssb'][cd.call_layer])
            if beta != 1.0:
                T20 = ((wt * wt).sum(1) / A32.shape[0])[:, None]                                    # the K = 0 (row-sum) form
                T2 = T20 + beta * (T2 - T20)
            return P1, T2

        def ss_star(w, A):
            P1, T2 = ss_factors(w.T, A if A.dtype == t.float32 else A.to(t.float32))
            result = ((T2 @ P1.T) / 3).to(t.float64).clone()
            result.diagonal().copy_((P1 * T2).sum(dim=1, dtype=t.float64))
            return result

        def ss_star_diagonal(w, A):
            P1, T2 = ss_factors(w.T, A.to(t.float32))
            return (P1 * T2).sum(dim=1, dtype=t.float64)
        cd.env['_flms_star'], cd.env['_flms_star_diagonal'] = ss_star, ss_star_diagonal
    if RDC is not None:      # comment READ COLUMN CUT
        from math import erfc, sqrt
        floor, sel1, read1 = 0.5 * erfc(float(RDC) / sqrt(2)), cd.select, cd.shared['read']

        def select_rd(score):      # the step's first selection (slot 0) is the live roster, select(gate)
            first = cd.slot == 0
            r = sel1(score)
            if first:
                side['rd_live'] = (r[0], int((score.detach() >= floor).sum()))
            return r
        cd.select = select_rd

        def read_rd(tt, pp, lam, response_indices, indices, U, F, old_columns):
            out = read1(tt, pp, lam, response_indices, indices, U, F, old_columns)
            live, n_floor = side['rd_live']
            assert t.equal(live, indices), 'the read columns are not the live roster'
            c = min(int(indices.numel()), ((n_floor + 31) // 32) * 32)
            if c >= int(indices.numel()):
                return out
            keep = t.ones(int(indices.numel()), dtype=out.dtype)
            keep[c:] = 0
            return out * keep[None, :]
        cd.shared['read'] = read_rd
    if EDGEX:      # comment EARLY EDGE
        assert not EDGEIN and EDGEXS is not None, "the early edge needs edgex_scales() and no edge inputs"
        lo, hi = EDGEX
        cov1, source = cd.hooks.get('cov_pm'), cd.env['_message_edge_source']

        def cov_early(l, C, ctx):
            C = C if cov1 is None else cov1(l, C, ctx)
            if not lo <= l <= hi or 8 <= l <= 14 or l < 1:
                return C
            context = {'left': cd.edge['context_left'], 'right': cd.edge['context_right']}
            return C + (EDGEXS[l] + eps['edx'][l]) * source(8, ctx['host'], ctx['k21'], ctx['aux'], ctx['w'], ctx['hstate'], cd.P, context)      # v5: + 'edx'
        cd.hooks['cov_pm'] = cov_early
