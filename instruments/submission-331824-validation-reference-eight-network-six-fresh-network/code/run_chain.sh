#!/usr/bin/env bash
# Serialized post-refit chain: wait for the terminal .pt, build the native candidate, paired-eval VALID40 (2 jobs) under a global lock, then gate/fire.
# Usage: run_chain.sh <pt_path> <kind: cumhead|export> <base_cand> <out_cand> <paired_json> <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; PT="$1"; KIND="$2"; BASE="$3"; OUT="$4"; PJ="$5"; TAG="$6"
PY=/Users/matt/anaconda3/bin/python; VENV="$HERE/../../../../../.venv/bin/python"; BANK="$HERE/../codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py"; LOCK=/tmp/fable_paired.lock
while [ ! -f "$PT" ]; do sleep 60; done; sleep 5
if [ "$KIND" = "cumhead" ]; then $PY "$HERE/build_cumhead_variant.py" "$BASE" "$PT" "$OUT" ${HEADS_ONLY:-}; else $PY "$HERE/export_params.py" "$BASE" "$PT" "$OUT"; fi || { echo "BUILD FAILED $TAG"; exit 1; }
while ! mkdir "$LOCK" 2>/dev/null; do sleep 30; done
nice -n 10 "$VENV" "$HERE/paired_eval.py" "$OUT" "$BANK" valid40 2 "$PJ"; rc=$?
rmdir "$LOCK"
[ $rc -eq 0 ] && bash "$HERE/gate_and_fire.sh" "$PJ" "$OUT" "$TAG"
