#!/usr/bin/env bash
# =============================================================================
# Thunders AI — Server Start Script
# =============================================================================
# Usage:
#   ./scripts/start_server.sh               # Production mode
#   ./scripts/start_server.sh --dev         # Development mode (auto-reload)
#   ./scripts/start_server.sh --port 8080   # Custom port
#   ./scripts/start_server.sh --gpu         # Enable GPU
#   ./scripts/start_server.sh --workers 4   # Number of workers
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
# Parse Arguments
# ---------------------------------------------------------------------------
MODE="prod"
HOST="0.0.0.0"
PORT=8000
WORKERS=1
DEVICE="cpu"
LOG_LEVEL="info"
RELOAD=false
CONFIG_FILE=""
ENABLE_METRICS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)          MODE="dev"; RELOAD=true; LOG_LEVEL="debug" ;;
        --host)         HOST="$2"; shift ;;
        --port)         PORT="$2"; shift ;;
        --workers)      WORKERS="$2"; shift ;;
        --gpu)          DEVICE="cuda" ;;
        --log-level)    LOG_LEVEL="$2"; shift ;;
        --config)       CONFIG_FILE="$2"; shift ;;
        --metrics)      ENABLE_METRICS=true ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev             Development mode (auto-reload, debug logging)"
            echo "  --host HOST       Bind host (default: 0.0.0.0)"
            echo "  --port PORT       Bind port (default: 8000)"
            echo "  --workers N       Number of Uvicorn workers (default: 1, prod: auto)"
            echo "  --gpu             Enable GPU/CUDA"
            echo "  --log-level LEVEL Logging level: debug|info|warning|error (default: info)"
            echo "  --config FILE     Path to config file"
            echo "  --metrics         Enable Prometheus metrics endpoint"
            echo "  -h, --help        Show this help message"
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
info "Thunders AI Server — $MODE mode"

# Check Python
if ! command -v python3 &>/dev/null; then
    error "Python 3 not found"
fi

# Check if thunders_ai is installed
if ! python3 -c "import thunders_ai" 2>/dev/null; then
    warn "thunders_ai not installed, installing in editable mode..."
    pip install -e "$PROJECT_ROOT" --quiet
    ok "thunders_ai installed"
fi

# Check for uvicorn
if ! python3 -c "import uvicorn" 2>/dev/null; then
    info "Installing uvicorn..."
    pip install uvicorn[standard] --quiet
    ok "uvicorn installed"
fi

# GPU check
if [[ "$DEVICE" == "cuda" ]]; then
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        ok "GPU: $GPU_NAME"
    else
        warn "CUDA requested but not available, falling back to CPU"
        DEVICE="cpu"
    fi
fi

# Auto-detect workers in production
if [[ "$MODE" == "prod" ]] && [[ "$WORKERS" -eq 1 ]]; then
    CPU_COUNT=$(python3 -c "import os; print(os.cpu_count() or 1)")
    WORKERS=$((CPU_COUNT > 4 ? 4 : CPU_COUNT))
    info "Auto-detected $WORKERS workers for production"
fi

# ---------------------------------------------------------------------------
# Set Environment
# ---------------------------------------------------------------------------
export THUNDERS_AI_HOST="$HOST"
export THUNDERS_AI_PORT="$PORT"
export THUNDERS_AI_DEVICE="$DEVICE"
export THUNDERS_AI_LOG_LEVEL="$LOG_LEVEL"
export THUNDERS_AI_WORKERS="$WORKERS"

if [[ -n "$CONFIG_FILE" ]]; then
    export THUNDERS_AI_CONFIG="$CONFIG_FILE"
fi

if [[ "$ENABLE_METRICS" == true ]]; then
    export THUNDERS_AI_METRICS="true"
fi

# ---------------------------------------------------------------------------
# Start Server
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
info "Starting Thunders AI Server"
echo "  Mode:     $MODE"
echo "  Host:     $HOST"
echo "  Port:     $PORT"
echo "  Workers:  $WORKERS"
echo "  Device:   $DEVICE"
echo "  Log:      $LOG_LEVEL"
echo "============================================"
echo ""

# Find the ASGI app
ASGI_APP="thunders_ai.api.app:app"

if [[ "$MODE" == "dev" ]]; then
    info "Development mode: auto-reload enabled"
    exec python3 -m uvicorn "$ASGI_APP" \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL" \
        --access-log
elif [[ "$MODE" == "prod" ]]; then
    info "Production mode: $WORKERS workers"
    exec python3 -m uvicorn "$ASGI_APP" \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --no-access-log \
        --loop uvloop \
        --http httptools
fi
