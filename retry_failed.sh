#!/usr/bin/env bash
# retry_failed.sh
#
# Retrains the 5 models that diverged in the previous sweep, with conservative
# per-model hyperparameters (lower LR + longer warmup) to avoid the gradient
# explosion that produced NaN losses last time.
#
# Appends results to the SAME training_results.csv as run_all_models.sh, so
# the downstream xlsx fill picks up the newer rows automatically (it dedupes
# by taking the LAST row per model).
#
# Usage (on the remote server):
#   cd ~/lmap_dev/pytorch-cifar100
#   chmod +x retry_failed.sh
#   tmux new -s retry
#   ./retry_failed.sh

set -u

# ------------------------- Configuration -------------------------
REPO_DIR="$HOME/lmap_dev/pytorch-cifar100"
CONDA_ENV="mncifar100"
GPUS="0,1"
BATCH_SIZE=256
SLEEP_SECONDS=300
LOG_DIR="$REPO_DIR/training_logs"
RESULTS_CSV="$REPO_DIR/training_results.csv"

# Per-model hyperparameters.  Format: "<model>:<warm>:<lr>"
# - preactresnet50/101 and attention56 each survived 40+ epochs before
#   diverging, so a moderate dial-down (LR 0.1->0.05, warm 1->10) should be
#   enough.
# - preactresnet152 and attention92 both diverged in epoch 1, so they need a
#   much more conservative schedule (LR 0.025, warm 20).
RUNS=(
  "preactresnet50:10:0.05"
  "preactresnet101:10:0.05"
  "preactresnet152:20:0.025"
  "attention56:10:0.05"
  "attention92:20:0.025"
)

# ------------------------- Conda activation -------------------------
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
  echo "ERROR: 'conda' not found." >&2
  exit 1
fi

conda activate "$CONDA_ENV" || {
  echo "ERROR: failed to activate conda env '$CONDA_ENV'." >&2
  exit 1
}

# ------------------------- Setup -------------------------
mkdir -p "$LOG_DIR"
cd "$REPO_DIR" || { echo "ERROR: $REPO_DIR not found" >&2; exit 1; }

# CSV header is expected to already exist (created by run_all_models.sh).
# We append new rows; downstream code dedupes by taking the last row per model.

total=${#RUNS[@]}

echo "============================================================"
echo "Retry sweep starting"
echo "  Models:    $total (the 5 that diverged)"
echo "  Repo:      $REPO_DIR"
echo "  Env:       $CONDA_ENV"
echo "  GPUs:      $GPUS"
echo "  Batch:     $BATCH_SIZE"
echo "  Logs:      $LOG_DIR  (per-model log will be OVERWRITTEN)"
echo "  Results:   $RESULTS_CSV  (rows APPENDED)"
echo "  Sleep:     ${SLEEP_SECONDS}s between models"
echo "============================================================"

# ------------------------- Main loop -------------------------
for i in "${!RUNS[@]}"; do
  IFS=':' read -r model warm lr <<< "${RUNS[$i]}"
  idx=$((i + 1))
  log_file="$LOG_DIR/${model}.log"
  started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  echo
  echo "-------- [$idx/$total] $model  warm=$warm  lr=$lr  started $started_at --------"

  CUDA_VISIBLE_DEVICES="$GPUS" python train.py \
      -net "$model" -gpu \
      -warm "$warm" \
      -b "$BATCH_SIZE" \
      -lr "$lr" \
    2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}

  finished_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

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

  if [[ $idx -lt $total ]]; then
    echo "Sleeping ${SLEEP_SECONDS}s before next model..."
    sleep "$SLEEP_SECONDS"
  fi
done

echo
echo "============================================================"
echo "Retry sweep done."
echo "Per-model logs:  $LOG_DIR/<model>.log"
echo "Summary CSV:     $RESULTS_CSV"
echo "============================================================"
