#!/bin/bash
# TAG opus_engineer a62 v1l (file written 2026-10-12 07:34Z) = port_rad_v1k.sh + K4L age 2 by layer (k4la2_from_pt_v1.py -> fragment_k4la2v_<name>.py;
# when non-empty, fragment_k4la2_v1.py + it are appended), family check v6.
# v1k header: (file written 2026-10-12 07:27Z) = port_rad_v1j.sh with k21n_from_pt_v2.py (the k21n table whenever any shift is non-zero:
# the v4 export leaves the knob at the bare base -0.028, trainer 07:2xZ).
# v1j header: (file written 2026-10-12 07:11Z) = port_rad_v1i.sh + k21n by layer (k21n_from_pt_v1.py -> fragment_k21nv_<name>.py; when
# non-empty, fragment_k21n_v1.py + it are appended) + rso / rsy (rs_from_pt_v1.py -> fragment_rsv_<name>.py; when non-empty,
# fragment_rs_v1.py + it are appended), family check v5. K21N_BASE=<V3 k21n> (env, default -0.028).
# v1i header: (file written 2026-10-12 04:29Z) = port_rad_v1h.sh (stack/s48_build_v2.sh) with the builder levers written at 12 significant digits (%.12g): S50's exact F4C / K4L values (%r, 17 digits each) made the source file name 259 bytes (build.py OSError; the stack limit is 244); 12 digits = a relative change below 1e-12, far under float32 resolution.
# v1h header: (file written 2026-10-12 04:15Z) = port_rad_v1g.sh with the kappa family DROPPED by default (KAPPA=carry to carry it):
# the route's parity at the torch arm's size FAILED (kap5 +0.05 at 1-14 vs a_RS50_fkappa_1_14 on 7100456 / 457: final-layer prediction
# change correlation 0.09 / 0.10, native norm 6.0 / 5.4 x torch), so the edit is not certified; S50's own shift (+0.0044) moves native
# predictions about 3e-7 of their size, below the native rounding floor (~2e-6 at L15): dropping it costs nothing measurable.
# v1g header: (file written 2026-10-12 03:55Z) = port_rad_v1f.sh + the 'kappa' family (S50's radial birth weight; v1f had no
# route and its family check stopped): kappa_from_pt_v1.py -> fragment_kappa_<name>.py; when non-zero, the release form's step birth line
# is edited by make_kappa_v1.py (s49/<name>_kp) and the S49 file is built on that; family check v4 (pt_families_v4.py, 'kappa' routed).
# v1f header: (2026-10-12 00:1xZ) = port_rad_v1e.sh + the 'k4' family (k4_from_pt_v1.py -> fragment_k4_<name>.py; with a
# non-zero shift, fragment_k4_v1.py + that file are appended: _S19_K4BAND = dials + band shifts, _K4_15 at the terminal map) and family
# check v3 (pt_families_v3.py, 'k4' routed).
# v1e header: port_rad_v1d.sh with the family check v2 (pt_families_v2.py: any family without a
# native route stops the port when non-zero, e.g. S50's 'k4').
# v1d header: port_rad_v1c.sh + K3L BY READ LAYER when nkl1-3 vary over layers 1-14 (law check by
# law_literals_from_pt_v2.py: _NP_K3L = layer 1; k3l_from_pt_v1.py -> fragment_k3l_<name>.py, appended with fragment_k3l_bylayer_v1.py).
# v1c header: port_rad_v1b.sh + the EARLY EDGE when the checkpoint carries non-zero 'edx' scales
# (edx_from_pt_v1.py -> fragment_edx_<name>.py; then fragment_early_edge_v1.py + that file are appended after the prebind list; with
# no edx the S49 file is v1b's) and the checkpoint's families listed against the MS point first (pt_families_v1.py; a FLAG stops).
# v1b header: port_rad_v1.sh with the source found by the build's own base token (BT@k<hex>) and
# the hand-over name; v1 took the newest file by mtime and, when the build refused to overwrite an existing source, picked another
# base's file (rehearsal on jmMS_v2: the HZ7Rg0.0 twin). An existing source is refused unless REUSE=1 (rehearsals only).
# v1 header: RAD-S48's port in one command, text only (nothing native runs).
# From the trainer's jmMS_v2-style hand-over O/trainer/candidates/<name>/ (candidate.py, physics_dials.json, per_layer_dials.json,
# head_weights_31x128.json) and its real checkpoint <pt>:
#   1. head: the hand-over's head table = jmNd's (cmp) -> jmNdh; otherwise converted to a42_head/weights/<name>h.json (+ _f15, + band)
#   2. source: BASE=s46h7rsd3x stack/s48_build.sh <name> <head> <pt>   (law literals, KC, per-layer dials from the checkpoint, head)
#   3. law literals: law_literals_from_pt_v1.py <pt> --check <source> (the source's six pair-law literals = the checkpoint's)
#   4. release form: stack/release_like.py <source> <name>_rl l3_bind128 lvfs_limit_off prices_zero
#   5. star strength: star_beta_from_pt_v1.py <pt> <name> -> fragment_star_beta_<name>.py (beta = 1 + ssb per layer)
#   6. S49: make_s49_v1.py rl/<name>_rl/candidate.py <name> + star_dir_v1 + star_beta_<name> + readcol_v1 (a >= -2.0) + prebind_s48b_v2
#      -> a62_modules/s49/<name>/candidate.py (= S48b's tail on the RAD point)
# Checks printed: the fragment count (5 = release-ready, FRAGS=none), the diff against S48b outside the learned values.
# usage: port_rad_v1.sh <hand-over name under O/trainer/candidates> <checkpoint .pt>
set -eu
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules
E=$(dirname $A); O=$(dirname $E); T=$O/trainer/candidates; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
name=$1; pt=$2
[ -d "$T/$name" ] && [ -f "$pt" ] || { echo "usage: port_rad_v1.sh <hand-over name> <checkpoint .pt>"; exit 2; }
[ ! -e "$A/s49/$name" ] && [ ! -e "$E/stack/rl/${name}_rl" ] || { echo "outputs for $name exist; pick a new name"; exit 2; }
echo "== $(date -u +%H:%M:%SZ) hand-over $T/$name, checkpoint $pt ($(shasum -a 256 "$pt" | cut -c1-8))"
$PY -B $A/pt_families_v6.py "$pt" $O/trainer/cert/RAD/MS/point_v1/params.pt | tee $A/port_${name}_families.txt | tail -3
grep -q "^VERDICT every per-layer family has a native route" $A/port_${name}_families.txt || { echo "a family has no native route: stop"; exit 5; }
if cmp -s "$T/$name/head_weights_31x128.json" "$T/jmNd/head_weights_31x128.json"; then
  head=jmNdh; echo "head table = jmNd's (byte for byte): jmNdh"
else
  head=${name}h
  [ -e "$E/a42_head/weights/$head.json" ] && { echo "a42_head/weights/$head.json exists; refusing to reuse it"; exit 3; }
  $PY -B $E/a42_head/weights_from_trainer.py --json "$T/$name/head_weights_31x128.json" all "$head" --candidate "$T/$name/candidate.py"
  echo "head table converted: $head"
fi
cd $E
log=$A/port_${name}_build.log
BASE=s46h7rsd3x bash $E/stack/s48_build_v2.sh "$name" "$head" "$pt" | tee $log
k=$(grep -o '^base k[0-9a-f]*' $log | tail -1 | cut -d' ' -f2)
[ -n "$k" ] || { echo "no base token in the build output"; exit 4; }
n=$(ls $E/stack/candidate_*_BT@${k}_*__"$name".py | wc -l)
[ "$n" -eq 1 ] || { echo "expected one source for BT@$k and $name, found $n"; exit 4; }
src=$(ls $E/stack/candidate_*_BT@${k}_*__"$name".py)
if grep -q 'AssertionError' $log; then
  [ "${REUSE:-0}" = 1 ] || { echo "the build refused (an existing source?): see $log; REUSE=1 only for rehearsals"; exit 4; }
  echo "REUSE=1: the existing source for BT@$k is taken"
fi
echo "source $(shasum -a 256 "$src" | cut -c1-8) $src"
$PY -B $A/law_literals_from_pt_v2.py "$pt" --check "$src" | tee $A/port_${name}_law.txt | tail -2
grep -q "^CHECK PASS" $A/port_${name}_law.txt || { echo "law literals differ from the checkpoint: stop"; exit 6; }
$PY -B $E/stack/release_like.py "$src" "${name}_rl" l3_bind128 lvfs_limit_off prices_zero > /dev/null
echo "release form $(shasum -a 256 $E/stack/rl/${name}_rl/candidate.py | cut -c1-8) stack/rl/${name}_rl/candidate.py"
$PY -B $A/star_beta_from_pt_v1.py "$pt" "$name"
$PY -B $A/edx_from_pt_v1.py "$pt" "$name"
ee=""
grep -q "^_EDX = {}$" $A/fragment_edx_${name}.py || ee="$A/fragment_early_edge_v1.py $A/fragment_edx_${name}.py"
[ -n "$ee" ] && echo "EARLY EDGE carried: $(grep "^_EDX" $A/fragment_edx_${name}.py)"
$PY -B $A/k3l_from_pt_v1.py "$pt" "$name" | tail -1
grep -q "^_NP_K3L_BY_LAYER = {}$" $A/fragment_k3l_${name}.py || { ee="$ee $A/fragment_k3l_bylayer_v1.py $A/fragment_k3l_${name}.py"; echo "K3L BY LAYER carried: $(grep "^_NP_K3L_BY_LAYER" $A/fragment_k3l_${name}.py | cut -c1-200)"; }
$PY -B $A/k4_from_pt_v1.py "$pt" "$name" | tail -2
grep -q "^_K4_FAMILY = True$" $A/fragment_k4_${name}.py && { ee="$ee $A/fragment_k4_v1.py $A/fragment_k4_${name}.py"; echo "K4 FAMILY carried: $(grep -E "^_S19_K4BAND|^_K4_15" $A/fragment_k4_${name}.py | tr "\n" " ")"; } || true
par=$E/stack/rl/${name}_rl/candidate.py
$PY -B $A/kappa_from_pt_v1.py "$pt" "$name" | tail -1
if ! grep -q "^_KAPPA = {}$" $A/fragment_kappa_${name}.py && [ "${KAPPA:-drop}" != carry ]; then
  echo "KAPPA DROPPED (uncertified route; KAPPA=carry carries it): $(grep "^_KAPPA" $A/fragment_kappa_${name}.py | cut -c1-120)"
fi
if ! grep -q "^_KAPPA = {}$" $A/fragment_kappa_${name}.py && [ "${KAPPA:-drop}" = carry ]; then
  $PY -B $A/make_kappa_v1.py "$par" "${name}_kp" $A/fragment_kappa_${name}.py
  par=$A/s49/${name}_kp/candidate.py; echo "KAPPA carried: $(grep "^_KAPPA" $A/fragment_kappa_${name}.py | cut -c1-120)"
fi
$PY -B $A/k21n_from_pt_v2.py "$pt" "$name" --base ${K21N_BASE:--0.028} | tail -1
grep -q "^_K21N_BY_LAYER = {}$" $A/fragment_k21nv_${name}.py || { ee="$ee $A/fragment_k21n_v1.py $A/fragment_k21nv_${name}.py"; echo "K21N BY LAYER carried: $(grep "^_K21N_BY_LAYER" $A/fragment_k21nv_${name}.py | cut -c1-160)"; }
$PY -B $A/rs_from_pt_v1.py "$pt" "$name" | tail -1
$PY -B $A/k4la2_from_pt_v1.py "$pt" "$name" | tail -1
grep -q "^_K4LA2_BY_LAYER = {}$" $A/fragment_k4la2v_${name}.py || { ee="$ee $A/fragment_k4la2_v1.py $A/fragment_k4la2v_${name}.py"; echo "K4L AGE 2 BY LAYER carried: $(grep "^_K4LA2_BY_LAYER" $A/fragment_k4la2v_${name}.py | cut -c1-160)"; }
grep -q "^_RS = {}$" $A/fragment_rsv_${name}.py || { ee="$ee $A/fragment_rs_v1.py $A/fragment_rsv_${name}.py"; echo "READ SCALES carried: $(grep "^_RS" $A/fragment_rsv_${name}.py | cut -c1-160)"; }
$PY -B $A/make_s49_v1.py "$par" "$name" $A/fragment_star_dir_v1.py $A/fragment_star_beta_${name}.py \
  $A/fragment_readcol_v1.py $A/fragment_prebind_s48b_v2.py $ee > /dev/null
f=$A/s49/$name/candidate.py
echo "S49 $(shasum -a 256 $f | cut -c1-8) $f; release fragment lines $(grep -c 'BEGIN OPUS-RELEASE FRAGMENT' $f) (5 = release-ready)"
echo "diff against S48b (8378970c), lines: $(diff $A/s48/s48b/candidate.py $f | grep -c '^[<>]' || true)"
