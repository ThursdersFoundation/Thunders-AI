#!/usr/bin/env bash
# =============================================================================
# Thunders AI — Installation Script
# =============================================================================
# Usage:
#   ./scripts/install.sh                    # Standard install
#   ./scripts/install.sh --gpu              # Install with GPU support
#   ./scripts/install.sh --dev              # Install with dev dependencies
#   ./scripts/install.sh --all              # Install all optional extras
#   ./scripts/install.sh --gpu --dev        # Combine options
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors & Logging
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Parse Arguments
# ---------------------------------------------------------------------------
INSTALL_GPU=false
INSTALL_DEV=false
INSTALL_ALL=false
INSTALL_VISION=false
INSTALL_SPEECH=false
INSTALL_ROBOTICS=false
SKIP_VENV=false
PYTHON_CMD="python3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       INSTALL_GPU=true ;;
        --dev)       INSTALL_DEV=true ;;
        --all)       INSTALL_ALL=true ;;
        --vision)    INSTALL_VISION=true ;;
        --speech)    INSTALL_SPEECH=true ;;
        --robotics)  INSTALL_ROBOTICS=true ;;
        --skip-venv) SKIP_VENV=true ;;
        --python)    PYTHON_CMD="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu         Install CUDA/GPU support"
            echo "  --dev         Install development dependencies"
            echo "  --all         Install all optional extras"
            echo "  --vision      Install vision AI dependencies"
            echo "  --speech      Install speech AI dependencies"
            echo "  --robotics    Install robotics dependencies"
            echo "  --skip-venv   Skip virtual environment creation"
            echo "  --python CMD  Use specific Python executable"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1. Use --help for usage."
            ;;
    esac
    shift
done

# If --all, enable everything
if [[ "$INSTALL_ALL" == true ]]; then
    INSTALL_GPU=true
    INSTALL_DEV=true
    INSTALL_VISION=true
    INSTALL_SPEECH=true
    INSTALL_ROBOTICS=true
fi

# ---------------------------------------------------------------------------
# Determine Project Root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

info "Thunders AI Installer"
info "Project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Check Python Version
# ---------------------------------------------------------------------------
info "Checking Python installation..."

if ! command -v "$PYTHON_CMD" &>/dev/null; then
    error "Python not found. Install Python 3.10+ and try again."
fi

PYTHON_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
    error "Python 3.10+ required. Found: $PYTHON_VERSION"
fi

ok "Python $PYTHON_VERSION detected"

# ---------------------------------------------------------------------------
# Create Virtual Environment
# ---------------------------------------------------------------------------
if [[ "$SKIP_VENV" == false ]]; then
    VENV_DIR="$PROJECT_ROOT/.venv"

    if [[ ! -d "$VENV_DIR" ]]; then
        info "Creating virtual environment at $VENV_DIR ..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        ok "Virtual environment created"
    else
        info "Virtual environment already exists at $VENV_DIR"
    fi

    # Activate venv
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="python"
    ok "Virtual environment activated"
fi

# Upgrade pip
info "Upgrading pip..."
"$PYTHON_CMD" -m pip install --upgrade pip setuptools wheel --quiet
ok "pip, setuptools, wheel upgraded"

# ---------------------------------------------------------------------------
# Install Base Dependencies
# ---------------------------------------------------------------------------
info "Installing base dependencies..."
"$PYTHON_CMD" -m pip install -r requirements/base.txt --quiet
ok "Base dependencies installed"

# ---------------------------------------------------------------------------
# Install Optional Extras
# ---------------------------------------------------------------------------
if [[ "$INSTALL_GPU" == true ]]; then
    info "Installing GPU dependencies..."
    if [[ -f requirements/gpu.txt ]]; then
        "$PYTHON_CMD" -m pip install -r requirements/gpu.txt --quiet
        ok "GPU dependencies installed"
    else
        warn "requirements/gpu.txt not found, installing torch with CUDA..."
        "$PYTHON_CMD" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
        ok "PyTorch with CUDA installed"
    fi
fi

if [[ "$INSTALL_DEV" == true ]]; then
    info "Installing development dependencies..."
    if [[ -f requirements/dev.txt ]]; then
        "$PYTHON_CMD" -m pip install -r requirements/dev.txt --quiet
        ok "Development dependencies installed"
    else
        warn "requirements/dev.txt not found, installing common dev tools..."
        "$PYTHON_CMD" -m pip install pytest pytest-cov black ruff mypy pre-commit --quiet
        ok "Dev tools installed"
    fi
fi

if [[ "$INSTALL_VISION" == true ]]; then
    info "Installing vision dependencies..."
    if [[ -f requirements/vision.txt ]]; then
        "$PYTHON_CMD" -m pip install -r requirements/vision.txt --quiet
    else
        "$PYTHON_CMD" -m pip install opencv-python pillow timm --quiet
    fi
    ok "Vision dependencies installed"
fi

if [[ "$INSTALL_SPEECH" == true ]]; then
    info "Installing speech dependencies..."
    if [[ -f requirements/speech.txt ]]; then
        "$PYTHON_CMD" -m pip install -r requirements/speech.txt --quiet
    else
        "$PYTHON_CMD" -m pip install openai-whisper pyttsx3 soundfile --quiet
    fi
    ok "Speech dependencies installed"
fi

if [[ "$INSTALL_ROBOTICS" == true ]]; then
    info "Installing robotics dependencies..."
    if [[ -f requirements/robotics.txt ]]; then
        "$PYTHON_CMD" -m pip install -r requirements/robotics.txt --quiet
    else
        "$PYTHON_CMD" -m pip install robotics pybullet gymnasium --quiet
    fi
    ok "Robotics dependencies installed"
fi

# ---------------------------------------------------------------------------
# Install Thunders AI Package (editable)
# ---------------------------------------------------------------------------
info "Installing Thunders AI package..."
"$PYTHON_CMD" -m pip install -e "$PROJECT_ROOT" --quiet
ok "Thunders AI package installed"

# ---------------------------------------------------------------------------
# Verify Installation
# ---------------------------------------------------------------------------
info "Verifying installation..."

VERIFICATION_PASSED=true

if "$PYTHON_CMD" -c "import thunders_ai" 2>/dev/null; then
    ok "thunders_ai module imported successfully"
else
    warn "Could not import thunders_ai module"
    VERIFICATION_PASSED=false
fi

if "$PYTHON_CMD" -c "from thunders_ai import __version__; print(__version__)" 2>/dev/null; then
    VERSION=$("$PYTHON_CMD" -c "from thunders_ai import __version__; print(__version__)")
    ok "Thunders AI version: $VERSION"
else
    warn "Could not determine version"
fi

if command -v thunders-ai &>/dev/null; then
    ok "CLI command 'thunders-ai' available"
else
    warn "CLI command 'thunders-ai' not on PATH (may need to activate venv)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
if [[ "$VERIFICATION_PASSED" == true ]]; then
    ok "Installation completed successfully!"
else
    warn "Installation completed with warnings"
fi
echo ""
echo "  Extras installed:"
echo "    GPU:      $INSTALL_GPU"
echo "    Dev:      $INSTALL_DEV"
echo "    Vision:   $INSTALL_VISION"
echo "    Speech:   $INSTALL_SPEECH"
echo "    Robotics: $INSTALL_ROBOTICS"
echo ""
echo "  Quick start:"
if [[ "$SKIP_VENV" == false ]]; then
    echo "    source .venv/bin/activate"
fi
echo "    thunders-ai serve"
echo "============================================"
