#!/usr/bin/env bash
# =============================================================================
# Thunders AI — Model Training Script
# =============================================================================
# Usage:
#   ./scripts/train_model.sh                              # Default training
#   ./scripts/train_model.sh --model llama --epochs 10    # Specific model
#   ./scripts/train_model.sh --distributed --gpus 4       # Distributed training
#   ./scripts/train_model.sh --resume checkpoint.pt       # Resume training
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors & Logging
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------
MODEL="default"
DATASET=""
EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
MAX_SEQ_LENGTH=512
GRADIENT_ACCUMULATION=1
FP16=false
BF16=false
DISTRIBUTED=false
NUM_GPUS=1
RESUME_FROM=""
OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
SEED=42
EVAL_INTERVAL=500
SAVE_INTERVAL=1000
WANDB=false
WANDB_PROJECT="thunders-ai-training"
CONFIG_FILE=""

# ---------------------------------------------------------------------------
# Parse Arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              MODEL="$2"; shift ;;
        --dataset)            DATASET="$2"; shift ;;
        --epochs)             EPOCHS="$2"; shift ;;
        --batch-size)         BATCH_SIZE="$2"; shift ;;
        --lr)                 LEARNING_RATE="$2"; shift ;;
        --weight-decay)       WEIGHT_DECAY="$2"; shift ;;
        --warmup)             WARMUP_STEPS="$2"; shift ;;
        --seq-length)         MAX_SEQ_LENGTH="$2"; shift ;;
        --grad-accum)         GRADIENT_ACCUMULATION="$2"; shift ;;
        --fp16)               FP16=true ;;
        --bf16)               BF16=true ;;
        --distributed)        DISTRIBUTED=true ;;
        --gpus)               NUM_GPUS="$2"; shift ;;
        --resume)             RESUME_FROM="$2"; shift ;;
        --output)             OUTPUT_DIR="$2"; shift ;;
        --log-dir)            LOG_DIR="$2"; shift ;;
        --seed)               SEED="$2"; shift ;;
        --eval-interval)      EVAL_INTERVAL="$2"; shift ;;
        --save-interval)      SAVE_INTERVAL="$2"; shift ;;
        --wandb)              WANDB=true ;;
        --wandb-project)      WANDB_PROJECT="$2"; shift ;;
        --config)             CONFIG_FILE="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Model & Data:"
            echo "  --model MODEL         Model architecture (default: default)"
            echo "  --dataset PATH        Training dataset path"
            echo "  --config FILE         Training config YAML/JSON"
            echo ""
            echo "Training:"
            echo "  --epochs N            Number of training epochs (default: 10)"
            echo "  --batch-size N        Batch size per GPU (default: 32)"
            echo "  --lr RATE             Learning rate (default: 3e-4)"
            echo "  --weight-decay W      Weight decay (default: 0.01)"
            echo "  --warmup N            Warmup steps (default: 500)"
            echo "  --seq-length N        Max sequence length (default: 512)"
            echo "  --grad-accum N        Gradient accumulation steps (default: 1)"
            echo "  --seed N              Random seed (default: 42)"
            echo ""
            echo "Precision & Distribution:"
            echo "  --fp16                Enable FP16 mixed precision"
            echo "  --bf16                Enable BF16 mixed precision"
            echo "  --distributed         Enable distributed training"
            echo "  --gpus N              Number of GPUs (default: 1)"
            echo ""
            echo "Checkpointing & Logging:"
            echo "  --resume PATH         Resume from checkpoint"
            echo "  --output DIR          Output directory (default: ./checkpoints)"
            echo "  --log-dir DIR         Log directory (default: ./logs)"
            echo "  --eval-interval N     Evaluation interval in steps (default: 500)"
            echo "  --save-interval N     Checkpoint save interval (default: 1000)"
            echo "  --wandb               Enable Weights & Biases logging"
            echo "  --wandb-project NAME  W&B project name (default: thunders-ai-training)"
            echo ""
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Determine Project Root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Pre-flight Checks
# ---------------------------------------------------------------------------
info "Thunders AI Training Pipeline"
info "Model: $MODEL | Epochs: $EPOCHS | Batch: $BATCH_SIZE | LR: $LEARNING_RATE"

# Check Python
if ! command -v python3 &>/dev/null; then
    error "Python 3 not found"
fi

# Check for PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    error "PyTorch not installed. Install with: pip install torch"
fi

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
info "PyTorch version: $TORCH_VERSION"

# GPU check
if [[ "$DISTRIBUTED" == true ]] || [[ "$NUM_GPUS" -gt 1 ]]; then
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        error "Distributed training requires CUDA but it's not available"
    fi
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    if [[ "$GPU_COUNT" -lt "$NUM_GPUS" ]]; then
        error "Requested $NUM_GPUS GPUs but only $GPU_COUNT available"
    fi
    ok "CUDA available with $GPU_COUNT GPU(s)"
fi

# Create output directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Set Environment for Distributed Training
# ---------------------------------------------------------------------------
if [[ "$DISTRIBUTED" == true ]]; then
    export MASTER_ADDR="localhost"
    export MASTER_PORT="29500"
    export WORLD_SIZE="$NUM_GPUS"
    info "Distributed training: $NUM_GPUS GPUs"
fi

# ---------------------------------------------------------------------------
# Build Training Command
# ---------------------------------------------------------------------------
TRAIN_CMD="python3"

if [[ "$DISTRIBUTED" == true ]]; then
    TRAIN_CMD="$TRAIN_CMD -m torch.distributed.launch --nproc_per_node=$NUM_GPUS"
fi

TRAIN_CMD="$TRAIN_CMD -m thunders_ai.train"

# Add arguments
TRAIN_CMD="$TRAIN_CMD \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --max-seq-length $MAX_SEQ_LENGTH \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --seed $SEED \
    --eval-interval $EVAL_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --output-dir $OUTPUT_DIR \
    --log-dir $LOG_DIR"

# Optional arguments
if [[ -n "$DATASET" ]]; then
    TRAIN_CMD="$TRAIN_CMD --dataset $DATASET"
fi

if [[ -n "$RESUME_FROM" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume-from $RESUME_FROM"
    info "Resuming from checkpoint: $RESUME_FROM"
fi

if [[ "$FP16" == true ]]; then
    TRAIN_CMD="$TRAIN_CMD --fp16"
    info "FP16 mixed precision enabled"
fi

if [[ "$BF16" == true ]]; then
    TRAIN_CMD="$TRAIN_CMD --bf16"
    info "BF16 mixed precision enabled"
fi

if [[ "$WANDB" == true ]]; then
    TRAIN_CMD="$TRAIN_CMD --wandb --wandb-project $WANDB_PROJECT"
    info "Weights & Biases logging enabled (project: $WANDB_PROJECT)"
fi

if [[ -n "$CONFIG_FILE" ]]; then
    TRAIN_CMD="$TRAIN_CMD --config $CONFIG_FILE"
    info "Using config file: $CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# Display Training Configuration
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
info "Training Configuration"
echo "============================================"
echo "  Model:              $MODEL"
echo "  Dataset:            ${DATASET:-'(default)'}"
echo "  Epochs:             $EPOCHS"
echo "  Batch Size:         $BATCH_SIZE"
echo "  Learning Rate:      $LEARNING_RATE"
echo "  Weight Decay:       $WEIGHT_DECAY"
echo "  Warmup Steps:       $WARMUP_STEPS"
echo "  Max Seq Length:     $MAX_SEQ_LENGTH"
echo "  Gradient Accum:     $GRADIENT_ACCUMULATION"
echo "  FP16:               $FP16"
echo "  BF16:               $BF16"
echo "  Distributed:        $DISTRIBUTED"
echo "  GPUs:               $NUM_GPUS"
echo "  Seed:               $SEED"
echo "  Output Dir:         $OUTPUT_DIR"
echo "  Log Dir:            $LOG_DIR"
echo "  Resume From:        ${RESUME_FROM:-'(none)'}"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Run Training
# ---------------------------------------------------------------------------
info "Starting training..."

START_TIME=$(date +%s)

# Run the training command
eval "$TRAIN_CMD"
TRAIN_EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "============================================"
    ok "Training completed successfully!"
    echo "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "  Checkpoints: $OUTPUT_DIR"
    echo "  Logs: $LOG_DIR"
    echo "============================================"
else
    echo ""
    error "Training failed with exit code $TRAIN_EXIT_CODE"
fi
