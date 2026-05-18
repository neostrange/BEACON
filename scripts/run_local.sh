#!/usr/bin/env bash
# BEACON – local WSL2 benchmark runner
# BEACON: Benchmark for Entity recognition Across Cybersecurity sources with unified ONtology
# Trains model candidates sequentially on GPU 0.
# A shared fixed split is created on the first run and reused by all subsequent runs.
#
# Usage:
#   bash scripts/run_local.sh              # full sweep
#   bash scripts/run_local.sh roberta-base # single model (pass model ID as $1)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_DIR/.venv"
DATASET="$REPO_DIR/dataset/beacon_stix_v1.csv"
OUTDIR="$REPO_DIR/outputs"
SPLIT_FILE="$OUTDIR/splits.json"   # shared across all runs

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTDIR"

# ── helper ───────────────────────────────────────────────────────────────────
run_model() {
    local model_id="$1"
    local batch="$2"
    local accum="$3"
    local lr="$4"
    local extra="${5:-}"

    local safe_name
    safe_name=$(echo "$model_id" | tr '/' '_')
    local log_file="$OUTDIR/${safe_name}.log"

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "  Training: $model_id"
    echo "  batch=$batch  accum=$accum  lr=$lr  eff_batch=$((batch * accum))"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # shellcheck disable=SC2086
    python "$REPO_DIR/scripts/train.py" \
        --dataset_path  "$DATASET"    \
        --model_type    "$model_id"   \
        --output_dir    "$OUTDIR"     \
        --split_file    "$SPLIT_FILE" \
        --batch_size    "$batch"      \
        --gradient_accumulation_steps "$accum" \
        --learning_rate "$lr"         \
        --lr_crf_fc     8e-5          \
        --max_seq_length 256          \
        --epochs         50           \
        --early_stopping_patience 8   \
        --warmup_proportion 0.1       \
        --dropout        0.2          \
        --seed           42           \
        --use_amp                     \
        $extra 2>&1 | tee -a "$log_file"

    echo "Finished: $model_id  →  $log_file"
}

# ── model shortlist ───────────────────────────────────────────────────────────
# Args: model_id  batch  grad_accum  lr  [extra_args]
#
# Stage A – base-class (comfortably fit 16 GB with AMP, batch 16)
declare -A MODELS
MODELS["roberta-base"]="16 1 3e-5"
MODELS["cisco-ai/SecureBERT2.0-base"]="16 1 3e-5"
MODELS["microsoft/deberta-v3-base"]="10 2 2e-5"
MODELS["answerdotai/ModernBERT-base"]="16 1 3e-5"
MODELS["ehsanaghaei/SecureBERT"]="16 1 3e-5"
MODELS["markusbayer/CySecBERT"]="16 1 3e-5"

# Ordered list (run roberta-base first as anchor)
ORDER=(
    "roberta-base"
    "cisco-ai/SecureBERT2.0-base"
    "microsoft/deberta-v3-base"
    "answerdotai/ModernBERT-base"
    "ehsanaghaei/SecureBERT"
    "markusbayer/CySecBERT"
)

# ── run ───────────────────────────────────────────────────────────────────────
TARGET="${1:-}"   # optional: single model override

if [[ -n "$TARGET" ]]; then
    # Run a single model passed on the command line
    if [[ -v "MODELS[$TARGET]" ]]; then
        read -r b a lr <<< "${MODELS[$TARGET]}"
        run_model "$TARGET" "$b" "$a" "$lr"
    else
        echo "Unknown model '$TARGET'. Add it to MODELS[] in this script."
        exit 1
    fi
else
    # Full sweep
    for m in "${ORDER[@]}"; do
        read -r b a lr <<< "${MODELS[$m]}"
        run_model "$m" "$b" "$a" "$lr"
    done
fi

echo ""
echo "All runs complete. Results are in $OUTDIR"
