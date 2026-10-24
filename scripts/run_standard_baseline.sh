#!/bin/bash
# Standard baseline (configs/pointer_baseline_std.yaml): old pointer agent under the plan v3 training setup.
# 5 folds x seeds 0,1,2 x budgets 4,6,8 on unstripped OASIS tokens, local window (rawL) first, then global (rawG).
# Per (tokens, fold, seed), budget 4 pretrains the 3 frozen readers and budgets 6 and 8 reuse them.
# Runs 3 (fold, seed) jobs at a time: 24 GB RAM, ~4.8 GB per run. Finished runs are skipped.
#
#     GPU=1 scripts/run_standard_baseline.sh            # whole queue
#     GPU=1 scripts/run_standard_baseline.sh rawL 0 0   # one (tokens, fold, seed) job
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then
  for v in rawL rawG; do for s in 0 1 2; do for f in 0 1 2 3 4; do echo $v $f $s; done; done; done | xargs -P 3 -L 1 "$0"
  echo QUEUE_DONE
  exit 0
fi
v=$1; f=$2; s=$3
case $v in rawG) td=data/oasis_tokens/neurovfm_tokens_raw_global;; rawL) td=data/oasis_tokens/neurovfm_tokens_raw_local_w2x4x4;; esac
readers=""
for b in 4 6 8; do
  name=stdbase_${v}_fold${f}_s${s}_b${b}
  if [ ! -f outputs/pointer_runs/$name/test_metrics.json ]; then
    rm -rf outputs/pointer_runs/$name
    CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python src/training/train_pointer_agent.py --config configs/pointer_baseline_std.yaml \
      --fold $f --seed $s --budget $b --token_dir $PWD/$td ${readers:+--readers $readers} --name $name > /dev/null 2>&1 \
      && echo "ok $name" || { echo "FAIL $name"; exit 1; }
  fi
  readers=outputs/pointer_runs/stdbase_${v}_fold${f}_s${s}_b4/readers.pt
done
