#!/usr/bin/env bash
# run_all_models.sh
#
# Sequentially trains each CIFAR-100 model in the MODELS array, sleeps
# SLEEP_SECONDS between runs, scrapes final accuracy + total training time
# from each model's log, and appends one row per model to a CSV.
#
# Usage (on the remote server):
#   chmod +x run_all_models.sh
#   nohup ./run_all_models.sh > run_all_models.out 2>&1 &      # background
#   # ...or run inside tmux/screen so an SSH drop doesn't kill it:
#   #   tmux new -s train
#   #   ./run_all_models.sh
#
# Resume: re-running the script will SKIP any model already recorded
# with status=ok in the CSV, so you can safely restart after a crash.

set -u

# ------------------------- Configuration -------------------------
REPO_DIR="$HOME/lmap_dev/pytorch-cifar100"
CONDA_ENV="mncifar100"
GPUS="0,1"
BATCH_SIZE=256
WARM=1
SLEEP_SECONDS=300                                # 5-minute cooldown between models
LOG_DIR="$REPO_DIR/training_logs"
RESULTS_CSV="$REPO_DIR/training_results.csv"

MODELS=(
  squeezenet mobilenet mobilenetv2 shufflenet shufflenetv2
  vgg11 vgg13 vgg16 vgg19
  densenet121 densenet161 densenet201
  googlenet inceptionv3 inceptionv4 inceptionresnetv2 xception
  resnet18 resnet34 resnet50 resnet101 resnet152
  preactresnet18 preactresnet34 preactresnet50 preactresnet101 preactresnet152
  resnext50 resnext101 resnext152
  attention56 attention92
  seresnet18 seresnet34 seresnet50 seresnet101 seresnet152
  nasnet wideresnet
  stochasticdepth18 stochasticdepth34 stochasticdepth50 stochasticdepth101
)

# ------------------------- Conda activation -------------------------
# Source conda's shell hook so `conda activate` works inside this script.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  for conda_sh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Edit this script to point at your conda install." >&2
  exit 1
fi

conda activate "$CONDA_ENV" || {
  echo "ERROR: failed to activate conda env '$CONDA_ENV'." >&2
  exit 1
}

# ------------------------- Setup -------------------------
mkdir -p "$LOG_DIR"
cd "$REPO_DIR" || { echo "ERROR: $REPO_DIR not found" >&2; exit 1; }

if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "model,accuracy,total_training_time_seconds,status,started_at,finished_at" > "$RESULTS_CSV"
fi

total=${#MODELS[@]}

echo "============================================================"
echo "Training sweep starting"
echo "  Models:    $total"
echo "  Repo:      $REPO_DIR"
echo "  Env:       $CONDA_ENV"
echo "  GPUs:      $GPUS"
echo "  Batch:     $BATCH_SIZE   Warm: $WARM"
echo "  Logs:      $LOG_DIR"
echo "  Results:   $RESULTS_CSV"
echo "  Sleep:     ${SLEEP_SECONDS}s between models"
echo "============================================================"

# ------------------------- Main loop -------------------------
for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  idx=$((i + 1))
  log_file="$LOG_DIR/${model}.log"

  # Skip if already completed successfully on a previous run.
  if grep -q "^${model},.*,ok," "$RESULTS_CSV" 2>/dev/null; then
    echo "[$idx/$total] $model  -- already done, skipping"
    continue
  fi

  started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  echo
  echo "-------- [$idx/$total] $model  started $started_at --------"

  # Run training; tee output so we both see it live AND save a log.
  # PIPESTATUS captures python's exit code even though we piped through tee.
  CUDA_VISIBLE_DEVICES="$GPUS" python train.py -net "$model" -gpu -warm "$WARM" -b "$BATCH_SIZE" \
    2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}

  finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  # Scrape the LAST occurrence of each pattern from the tail of the log.
  #   Expected:  "Accuracy: 0.7543"
  #              "Total training time: 2372.1234567891234s"
  accuracy=$(tail -n 400 "$log_file" \
               | grep -oE 'Accuracy:[[:space:]]*[0-9.]+' \
               | tail -n1 \
               | awk '{print $2}')
  total_time=$(tail -n 400 "$log_file" \
                 | grep -oE 'Total training time:[[:space:]]*[0-9.]+s?' \
                 | tail -n1 \
                 | sed -E 's/[^0-9.]//g')

  if [[ $status -eq 0 ]]; then
    run_status="ok"
    echo ">>> [$idx/$total] $model DONE  acc=${accuracy:-?}  time=${total_time:-?}s"
  else
    run_status="failed_exit_${status}"
    echo ">>> [$idx/$total] $model FAILED (exit $status)"
  fi

  echo "${model},${accuracy:-},${total_time:-},${run_status},${started_at},${finished_at}" >> "$RESULTS_CSV"

  # Cooldown (skip after the final model).
  if [[ $idx -lt $total ]]; then
    echo "Sleeping ${SLEEP_SECONDS}s before next model..."
    sleep "$SLEEP_SECONDS"
  fi
done

echo
echo "============================================================"
echo "All done."
echo "Per-model logs:  $LOG_DIR/<model>.log"
echo "Summary CSV:     $RESULTS_CSV"
echo "============================================================"
