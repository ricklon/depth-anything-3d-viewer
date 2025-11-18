# Download metric depth checkpoints for Depth-Anything-3D

Write-Host "Downloading metric depth checkpoints..." -ForegroundColor Green

# Create checkpoints directory
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

# Base URL
$baseUrl = "https://huggingface.co/depth-anything"

# Checkpoint files - CORRECTED URLs for Metric models
$checkpoints = @(
    @{
        Name = "Small (vits) - 28.4M params - Recommended for webcam"
        File = "metric_video_depth_anything_vits.pth"
        Url = "$baseUrl/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth"
    },
    @{
        Name = "Base (vitb) - 113.1M params"
        File = "metric_video_depth_anything_vitb.pth"
        Url = "$baseUrl/Metric-Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth"
    },
    @{
        Name = "Large (vitl) - 381.8M params - Best quality"
        File = "metric_video_depth_anything_vitl.pth"
        Url = "$baseUrl/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth"
    }
)

# Ask which to download
Write-Host "`nAvailable checkpoints:" -ForegroundColor Cyan
for ($i = 0; $i -lt $checkpoints.Count; $i++) {
    Write-Host "  [$($i+1)] $($checkpoints[$i].Name)"
}
Write-Host "  [4] Download all" -ForegroundColor Yellow

$choice = Read-Host "`nEnter choice (1-4)"

$toDownload = @()
switch ($choice) {
    "1" { $toDownload = @($checkpoints[0]) }
    "2" { $toDownload = @($checkpoints[1]) }
    "3" { $toDownload = @($checkpoints[2]) }
    "4" { $toDownload = $checkpoints }
    default {
        Write-Host "Invalid choice. Downloading Small (recommended)..." -ForegroundColor Yellow
        $toDownload = @($checkpoints[0])
    }
}

# Download selected checkpoints
foreach ($checkpoint in $toDownload) {
    $outPath = "checkpoints\$($checkpoint.File)"

    # Check if already exists
    if (Test-Path $outPath) {
        Write-Host "`n✓ $($checkpoint.File) already exists, skipping..." -ForegroundColor Green
        continue
    }

    Write-Host "`nDownloading $($checkpoint.Name)..." -ForegroundColor Cyan
    Write-Host "  URL: $($checkpoint.Url)" -ForegroundColor Gray
    Write-Host "  Saving to: $outPath" -ForegroundColor Gray

    try {
        # Use curl instead of Invoke-WebRequest (more reliable)
        curl.exe -L -o $outPath $($checkpoint.Url)

        if (Test-Path $outPath) {
            $size = (Get-Item $outPath).Length / 1MB
            Write-Host "✓ Downloaded successfully! Size: $([math]::Round($size, 1)) MB" -ForegroundColor Green
        } else {
            Write-Host "✗ Download failed!" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ Error: $_" -ForegroundColor Red
    }
}

Write-Host "`n=== Download Complete ===" -ForegroundColor Green
Write-Host "`nYou can now run:" -ForegroundColor Cyan
Write-Host "  uv run da3d webcam3d --metric --encoder vits" -ForegroundColor Yellow
