#!/usr/bin/env bash
# =============================================================================
# Thunders AI — Build Script
# =============================================================================
# Usage:
#   ./scripts/build.sh                  # Full build (clean + test + package)
#   ./scripts/build.sh --skip-tests     # Skip test suite
#   ./scripts/build.sh --skip-clean     # Skip clean step
#   ./scripts/build.sh --wheel-only     # Only build wheel
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
SKIP_TESTS=false
SKIP_CLEAN=false
WHEEL_ONLY=false
BUILD_DOCS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-tests)  SKIP_TESTS=true ;;
        --skip-clean)  SKIP_CLEAN=true ;;
        --wheel-only)  WHEEL_ONLY=true ;;
        --docs)        BUILD_DOCS=true ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-tests   Skip running test suite"
            echo "  --skip-clean   Skip clean step"
            echo "  --wheel-only   Only build wheel, no source dist"
            echo "  --docs         Build documentation"
            echo "  -h, --help     Show this help message"
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

BUILD_DIR="$PROJECT_ROOT/dist"
DIST_DIR="$PROJECT_ROOT/dist"

info "Thunders AI Build System"
info "Project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Step 1: Clean Previous Builds
# ---------------------------------------------------------------------------
if [[ "$SKIP_CLEAN" == false ]]; then
    info "Cleaning previous build artifacts..."

    # Remove dist/ and build/
    rm -rf "$PROJECT_ROOT/dist" "$PROJECT_ROOT/build"

    # Remove egg-info directories
    find "$PROJECT_ROOT" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

    # Remove __pycache__ directories
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    # Remove .pyc files
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true

    # Remove .pytest_cache
    rm -rf "$PROJECT_ROOT/.pytest_cache" 2>/dev/null || true

    # Remove .mypy_cache
    rm -rf "$PROJECT_ROOT/.mypy_cache" 2>/dev/null || true

    ok "Build artifacts cleaned"
else
    warn "Skipping clean step"
fi

# ---------------------------------------------------------------------------
# Step 2: Run Tests
# ---------------------------------------------------------------------------
if [[ "$SKIP_TESTS" == false ]]; then
    info "Running test suite..."

    # Check for pytest
    if command -v pytest &>/dev/null; then
        pytest "$PROJECT_ROOT/tests/" \
            --tb=short \
            --no-header \
            -q \
            || error "Tests failed! Fix issues before building."

        ok "All tests passed"
    else
        warn "pytest not found, skipping tests. Install with: pip install pytest"
    fi
else
    warn "Skipping test suite"
fi

# ---------------------------------------------------------------------------
# Step 3: Type Check (optional but recommended)
# ---------------------------------------------------------------------------
if command -v mypy &>/dev/null; then
    info "Running type checks..."
    mypy "$PROJECT_ROOT/thunders_ai/" \
        --ignore-missing-imports \
        --no-error-summary \
        --quiet \
        2>/dev/null || warn "Type check warnings found (non-blocking)"
    ok "Type check completed"
else
    warn "mypy not found, skipping type checks"
fi

# ---------------------------------------------------------------------------
# Step 4: Lint (optional but recommended)
# ---------------------------------------------------------------------------
if command -v ruff &>/dev/null; then
    info "Running linter..."
    ruff check "$PROJECT_ROOT/thunders_ai/" --quiet || warn "Lint warnings found (non-blocking)"
    ok "Lint check completed"
else
    warn "ruff not found, skipping lint checks"
fi

# ---------------------------------------------------------------------------
# Step 5: Build Package
# ---------------------------------------------------------------------------
info "Building package..."

# Ensure build tool is available
if ! python3 -c "import build" 2>/dev/null; then
    info "Installing build tool..."
    pip install build --quiet
fi

# Build
if [[ "$WHEEL_ONLY" == true ]]; then
    python3 -m build --wheel "$PROJECT_ROOT"
else
    python3 -m build "$PROJECT_ROOT"
fi

ok "Package built successfully"

# ---------------------------------------------------------------------------
# Step 6: Verify Build Artifacts
# ---------------------------------------------------------------------------
info "Verifying build artifacts..."

if [[ ! -d "$DIST_DIR" ]]; then
    error "dist/ directory not found after build"
fi

WHEEL_COUNT=$(find "$DIST_DIR" -name "*.whl" | wc -l)
SDIST_COUNT=$(find "$DIST_DIR" -name "*.tar.gz" | wc -l)

if [[ "$WHEEL_COUNT" -eq 0 ]]; then
    error "No wheel file found in dist/"
fi

ok "Found $WHEEL_COUNT wheel(s) and $SDIST_COUNT source distribution(s)"

# List artifacts
echo ""
info "Build artifacts:"
ls -lh "$DIST_DIR/"

# Verify wheel is installable
info "Verifying wheel is installable..."
WHEEL_FILE=$(find "$DIST_DIR" -name "*.whl" -print -quit)
if python3 -m pip install --dry-run "$WHEEL_FILE" &>/dev/null; then
    ok "Wheel is installable"
else
    warn "Wheel install check failed (may need missing system deps)"
fi

# Check wheel contents
info "Wheel contents:"
python3 -m zipfile -l "$WHEEL_FILE" | head -20

# ---------------------------------------------------------------------------
# Step 7: Build Documentation (optional)
# ---------------------------------------------------------------------------
if [[ "$BUILD_DOCS" == true ]]; then
    info "Building documentation..."
    if [[ -f "$PROJECT_ROOT/docs/Makefile" ]]; then
        (cd "$PROJECT_ROOT/docs" && make html)
        ok "Documentation built"
    else
        warn "No docs/Makefile found, skipping documentation build"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
ok "Build completed successfully!"
echo ""
echo "  Artifacts:"
ls -1 "$DIST_DIR/" | while read -r f; do
    echo "    - $f"
done
echo ""
echo "  Install locally:"
echo "    pip install $DIST_DIR/*.whl"
echo ""
echo "  Upload to PyPI:"
echo "    twine upload $DIST_DIR/*"
echo "============================================"
