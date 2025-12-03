#!/bin/bash
# Test script to verify all features have required assets and can run

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================="
echo "Depth-Anything-3D Feature Test Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to test command existence
test_command() {
    local cmd=$1
    local description=$2

    echo -n "Testing: $description... "

    if uv run da3d $cmd --help &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to check file exists
check_file() {
    local file=$1
    local description=$2

    echo -n "Checking: $description... "

    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ EXISTS${NC} ($file)"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ MISSING${NC} ($file)"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to check optional file
check_optional_file() {
    local file=$1
    local description=$2

    echo -n "Checking (optional): $description... "

    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ EXISTS${NC} ($file)"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${YELLOW}⊘ OPTIONAL${NC} ($file)"
        ((TESTS_SKIPPED++))
        return 1
    fi
}

echo "=== Command Tests ==="
echo ""

test_command "webcam3d" "webcam3d command"
test_command "screen3d-viewer" "screen3d-viewer command"
test_command "view3d" "view3d command"
test_command "screen3d" "screen3d command"
test_command "video" "video command"
test_command "webcam" "webcam command"
test_command "demo" "demo command"
test_command "projector-preview" "projector-preview command"
test_command "projector-calibrate" "projector-calibrate command"

echo ""
echo "=== Required Model Checkpoints ==="
echo ""

check_file "checkpoints/video_depth_anything_vits.pth" "VDA Small model (required)"

echo ""
echo "=== Optional Model Checkpoints ==="
echo ""

check_optional_file "checkpoints/video_depth_anything_vitb.pth" "VDA Base model"
check_optional_file "checkpoints/video_depth_anything_vitl.pth" "VDA Large model"
check_optional_file "checkpoints/metric_video_depth_anything_vits.pth" "Metric VDA Small model"

echo ""
echo "=== Test Assets ==="
echo ""

check_file "tests/data/test_image.jpg" "Test image"
check_file "tests/data/test_depth.png" "Test depth PNG"
check_file "tests/data/test_depth.npy" "Test depth NPY"

echo ""
echo "=== Projector Assets ==="
echo ""

check_file "assets/test_pattern.png" "Test pattern"
check_file "assets/lobby_cube.obj" "Lobby cube 3D model"
check_file "config/projection_example.yaml" "Projection config"

echo ""
echo "=== Configuration Files ==="
echo ""

check_file "config.yaml" "Main configuration"
check_file "pyproject.toml" "Package configuration"

echo ""
echo "=== Documentation ==="
echo ""

check_file "README.md" "README"
check_file "CLAUDE.md" "CLAUDE.md"
check_file "MODELS.md" "Models guide"
check_file "FEATURE_ASSET_AUDIT.md" "Asset audit"

echo ""
echo "========================================="
echo "Test Results Summary"
echo "========================================="
echo -e "${GREEN}Passed:${NC}  $TESTS_PASSED"
echo -e "${YELLOW}Skipped:${NC} $TESTS_SKIPPED (optional)"
echo -e "${RED}Failed:${NC}  $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All required tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: uv run da3d view3d tests/data/test_image.jpg tests/data/test_depth.png"
    echo "  2. Test projector: uv run da3d projector-preview --config config/projection_example.yaml --show test_pattern_show"
    echo "  3. See MODELS.md for downloading optional models"
    exit 0
else
    echo -e "${RED}✗ Some required tests failed${NC}"
    echo ""
    echo "Please fix the failed items before using the package."
    echo "See MODELS.md for model downloads."
    echo "See FEATURE_ASSET_AUDIT.md for complete asset list."
    exit 1
fi
