#!/bin/bash
#
# Setup script for Video-Depth-Anything dependency
#
# This script clones and sets up the Video-Depth-Anything repository
# which is required for depth-anything-3d to function.

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
VIDEO_DEPTH_DIR="$PARENT_DIR/Video-Depth-Anything"

echo "========================================"
echo "Video-Depth-Anything Setup Script"
echo "========================================"
echo ""

# Check if already exists
if [ -d "$VIDEO_DEPTH_DIR" ]; then
    echo "✓ Video-Depth-Anything directory already exists at:"
    echo "  $VIDEO_DEPTH_DIR"
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating Video-Depth-Anything..."
        cd "$VIDEO_DEPTH_DIR"
        git pull
    fi
else
    echo "Cloning Video-Depth-Anything repository..."
    cd "$PARENT_DIR"
    git clone https://github.com/DepthAnything/Video-Depth-Anything
    echo "✓ Cloned successfully"
fi

echo ""
echo "Installing Video-Depth-Anything dependencies..."
cd "$VIDEO_DEPTH_DIR"

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "⚠ Warning: requirements.txt not found"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Video-Depth-Anything is now available at:"
echo "  $VIDEO_DEPTH_DIR"
echo ""
echo "Next steps:"
echo "1. Download model checkpoints:"
echo "   cd $VIDEO_DEPTH_DIR/checkpoints"
echo "   wget https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
echo ""
echo "2. Test the installation:"
echo "   da3d --help"
echo ""
