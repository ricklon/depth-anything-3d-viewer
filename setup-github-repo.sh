#!/bin/bash
# GitHub Repository Setup Script for depth-anything-3d-viewer
# Uses GitHub CLI (gh) to create and configure the repository

set -e  # Exit on error

echo "=========================================="
echo "Depth-Anything-3D GitHub Repo Setup"
echo "=========================================="
echo ""

# Navigate to the project directory
cd "$(dirname "$0")"

echo "📂 Current directory: $(pwd)"
echo ""

# Check if gh CLI is installed and authenticated
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed!"
    echo "Install from: https://cli.github.com/"
    exit 1
fi

echo "✅ GitHub CLI found"
echo ""

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI"
    echo "Run: gh auth login"
    exit 1
fi

echo "✅ Authenticated with GitHub"
echo ""

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "📝 Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi
echo ""

# Add all files
echo "📝 Adding files to git..."
git add .
echo "✅ Files added"
echo ""

# Create initial commit
echo "📝 Creating initial commit..."
if git diff --cached --quiet; then
    echo "⚠️  No changes to commit (already committed)"
else
    git commit -m "Initial commit: Depth-Anything-3D viewer v0.1.0

Features:
- Real-time 3D webcam visualization (webcam3d)
- Real-time 3D screen capture (screen3d-viewer)
- Static 3D depth map viewing (view3d)
- 2.5D parallax effects (screen3d)
- Mesh and point cloud visualization modes
- Proportional Z-depth scaling
- Percentile-based depth clamping
- Performance tuning options
- Virtual camera output for OBS
- Comprehensive CLI and Python API"
    echo "✅ Initial commit created"
fi
echo ""

# Create GitHub repository
echo "📝 Creating GitHub repository..."
gh repo create ricklon/depth-anything-3d-viewer \
    --public \
    --source=. \
    --description="Interactive 3D mesh visualization and real-time rendering for Video-Depth-Anything depth maps" \
    --push

echo "✅ Repository created and pushed!"
echo ""

# Add topics
echo "📝 Adding repository topics..."
gh repo edit ricklon/depth-anything-3d-viewer \
    --add-topic depth-estimation \
    --add-topic 3d-visualization \
    --add-topic computer-vision \
    --add-topic depth-anything \
    --add-topic real-time-rendering \
    --add-topic open3d \
    --add-topic pytorch \
    --add-topic python \
    --add-topic 3d-mesh \
    --add-topic point-cloud

echo "✅ Topics added"
echo ""

# Enable discussions
echo "📝 Enabling discussions..."
gh repo edit ricklon/depth-anything-3d-viewer --enable-discussions || echo "⚠️  Could not enable discussions (may need manual setup)"
echo ""

# Enable issues (should be on by default)
echo "📝 Enabling issues..."
gh repo edit ricklon/depth-anything-3d-viewer --enable-issues
echo "✅ Issues enabled"
echo ""

echo "=========================================="
echo "✨ Repository setup complete!"
echo "=========================================="
echo ""
echo "🔗 Repository URL: https://github.com/ricklon/depth-anything-3d-viewer"
echo ""
echo "Next steps:"
echo "1. Visit your repository: gh repo view ricklon/depth-anything-3d-viewer --web"
echo "2. Create a release: gh release create v0.1.0 --title 'v0.1.0 - Initial Release' --notes-file CHANGELOG.md"
echo "3. Add repository description and website in GitHub settings"
echo ""
echo "Optional:"
echo "- Set up branch protection rules"
echo "- Configure GitHub Actions for CI/CD"
echo "- Add screenshots/demo GIFs to README"
echo ""
