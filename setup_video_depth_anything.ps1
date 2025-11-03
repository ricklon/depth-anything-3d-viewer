# PowerShell setup script for Video-Depth-Anything dependency
#
# This script clones and sets up the Video-Depth-Anything repository
# which is required for depth-anything-3d to function.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentDir = Split-Path -Parent $ScriptDir
$VideoDepthDir = Join-Path $ParentDir "Video-Depth-Anything"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Video-Depth-Anything Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if already exists
if (Test-Path $VideoDepthDir) {
    Write-Host "✓ Video-Depth-Anything directory already exists at:" -ForegroundColor Green
    Write-Host "  $VideoDepthDir"
    Write-Host ""
    $response = Read-Host "Do you want to update it? (y/n)"
    if ($response -match "^[Yy]$") {
        Write-Host "Updating Video-Depth-Anything..." -ForegroundColor Yellow
        Push-Location $VideoDepthDir
        git pull
        Pop-Location
    }
} else {
    Write-Host "Cloning Video-Depth-Anything repository..." -ForegroundColor Yellow
    Push-Location $ParentDir
    git clone https://github.com/DepthAnything/Video-Depth-Anything
    Pop-Location
    Write-Host "✓ Cloned successfully" -ForegroundColor Green
}

Write-Host ""
Write-Host "Installing Video-Depth-Anything dependencies..." -ForegroundColor Yellow
Push-Location $VideoDepthDir

if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: requirements.txt not found" -ForegroundColor Yellow
}

Pop-Location

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Video-Depth-Anything is now available at:" -ForegroundColor White
Write-Host "  $VideoDepthDir" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Download model checkpoints:" -ForegroundColor White
Write-Host "   cd $VideoDepthDir\checkpoints" -ForegroundColor Gray
Write-Host "   Invoke-WebRequest -Uri 'https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth' -OutFile 'video_depth_anything_vits.pth'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Test the installation:" -ForegroundColor White
Write-Host "   da3d --help" -ForegroundColor Gray
Write-Host ""
