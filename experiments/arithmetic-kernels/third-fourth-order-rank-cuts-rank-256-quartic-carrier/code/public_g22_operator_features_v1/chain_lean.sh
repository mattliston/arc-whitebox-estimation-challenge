#!/bin/bash
# TAG g22-chain-lean (T-118i): co-adapt the closure to the LEAN cards (full CP3 bank): LEAN-2 (CP4 off + no quartic) then LEAN (+ no cubic);
# rows -> train (init HNQ) -> loops with util -> 1e9 ranking -> gate: if raw x util (8M) < 0.9e-8 on both nets, build/score/panel/fire.
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; REPO=/Users/matt/arc-white-box-estimation; cd $D
run_round() {   # $1 name, $2 env string for the card, $3 build env string
  local NAME=$1; local ENVS=$2; local BENV=$3
  while pgrep -f "gen_rows_meanleak_mech15.*py [0-9]" > /dev/null; do sleep 60; done
  env $ENVS bash -c "sed 's#rows_meanleak_mech15_cap#rows_meanleak_mech15_$NAME#g' launch_mech15_cap.sh > launch_mech15_$NAME.sh; chmod +x launch_mech15_$NAME.sh; ./launch_mech15_$NAME.sh"
  sleep 120
  while [ "$(ls -d rows_meanleak_mech15_$NAME/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
  while pgrep -f "gen_rows_meanleak_mech15_cap.py [0-9]" > /dev/null; do sleep 20; done
  GNN_FIRST=1 GNN_INIT=GNN_GNN_HNQ_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 $AP train_gnn_oracle.py $D/rows_meanleak_mech15_$NAME GNN_H$NAME 30 64 2>&1 | grep -v Warning > train_gnn_h$NAME.log
  grep -E "^epoch (10|30)" train_gnn_h$NAME.log | cut -c1-200
  EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_H${NAME}_MODEL.pt payload_gnn_h$NAME.npz 2>&1 | grep -v Warning | cut -c1-80
  local l35 l100 u35 u100
  for r in 35 100; do
    env $ENVS RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_h$NAME.npz TAG=closedloop_H$NAME THREADS=4 \
      $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"u_local": [0-9.]*\|"mse_nc_f2j_off": [0-9.e-]*' | tr '\n' ' ' | sed "s/^/r$r closedloop_H$NAME (co-adapted $NAME) CLOSED LOOP: /"; echo
  done
  $PY score_vs_truthpost.py closedloop_H$NAME closedloop_HNQ closedloop_HV_mix0.0
  l35=$(grep "^r35 closedloop_H$NAME" chain_lean.log | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | grep -o "[0-9.e-]*$"); u35=$(grep "^r35 closedloop_H$NAME" chain_lean.log | grep -o '"u_local": [0-9.]*' | grep -o "[0-9.]*$")
  l100=$(grep "^r100 closedloop_H$NAME" chain_lean.log | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | grep -o "[0-9.e-]*$"); u100=$(grep "^r100 closedloop_H$NAME" chain_lean.log | grep -o '"u_local": [0-9.]*' | grep -o "[0-9.]*$")
  echo "$NAME raw x util (8M): r35 $($PY -c "print(float('${l35:-1}')*float('${u35:-1}'))") r100 $($PY -c "print(float('${l100:-1}')*float('${u100:-1}'))")"
  if $PY -c "import sys; sys.exit(0 if float('${l35:-1}')*float('${u35:-1}') < 0.9e-8 and float('${l100:-1}')*float('${u100:-1}') < 0.9e-8 else 1)"; then
    echo "$NAME passes raw x util < 0.9e-8 on both nets -> candidate"
    env $BENV $AP build_gnn_candidate.py $D/payload_gnn_h$NAME.npz $D/cand_gnn_$NAME.py 2>&1 | grep -v Warning | tail -1 | cut -c1-120
    (cd $REPO && $HOME/.local/bin/uv run whest validate --estimator $D/cand_gnn_$NAME.py 2>&1 | tail -1)
    for r in 35 100; do nice -n 10 $PY score_candidate.py $D/cand_gnn_$NAME.py $r 2>&1 | grep -v Warning | tail -1 | cut -c1-200; done
    a35=$(grep "cand_gnn_$NAME.py\", \"rank\": 35" chain_lean.log | tail -1 | grep -o '"adjusted": [0-9.e-]*' | grep -o "[0-9.e-]*$"); a100=$(grep "cand_gnn_$NAME.py\", \"rank\": 100" chain_lean.log | tail -1 | grep -o '"adjusted": [0-9.e-]*' | grep -o "[0-9.e-]*$")
    if $PY -c "import sys; sys.exit(0 if float('${a35:-1}') < 1e-8 and float('${a100:-1}') < 1e-8 else 1)"; then
      echo "$NAME candidate adjusted < 1e-8 on both nets -> PANEL GATE"; ./panel_candidate.sh $D/cand_gnn_$NAME.py gnn_$NAME > panel_summary_gnn_$NAME.txt 2>&1; cat panel_summary_gnn_$NAME.txt
      P8=$(grep -o "panel8: .*mean adjusted [0-9.e-]*" panel_summary_gnn_$NAME.txt | grep -o "[0-9.e-]*$"); V6=$(grep -o "virgin6: .*mean adjusted [0-9.e-]*" panel_summary_gnn_$NAME.txt | grep -o "[0-9.e-]*$")
      if $PY -c "import sys; sys.exit(0 if float('${P8:-1}') < 1e-8 and float('${V6:-1}') < 1e-8 else 1)"; then
        echo "GATE PASSED (panel8 $P8, virgin6 $V6) -> submitting (preauthorized) $(date)"
        (cd $REPO && bash autoresearch/submit_cand.sh $D/cand_gnn_$NAME.py "T-118 lean card + learned closure ($NAME): panel8 adj $P8, virgin6 adj $V6" 2>&1 | tail -5); echo "SUBMITTED $NAME $(date)"
      else echo "GATE NOT PASSED for $NAME"; fi
    else echo "$NAME candidate not under 1e-8 adjusted in the grader frame ($a35 / $a100)"; fi
  fi
}
run_round lean2 "CP4_CAP=1 RETRO_NO_QUARTIC=1" "BUILD_CP4_CAP=1 BUILD_NO_QUARTIC=1"
run_round lean "CP4_CAP=1 RETRO_NO_QUARTIC=1 RETRO_NO_CUBIC=1" "BUILD_CP4_CAP=1 BUILD_NO_QUARTIC=1 BUILD_NO_CUBIC=1"
echo LEAN_DONE
