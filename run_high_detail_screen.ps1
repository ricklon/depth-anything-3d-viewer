# Run screen3d-viewer with the "Sweet Spot" high-detail settings
# Based on user testing, 960p with 2x subsampling provides a great balance of detail and performance.
# Uses the default Small (vits) model.

Write-Host "Starting High-Detail 3D Screen Viewer..."
Write-Host "Settings: Resolution=960px, Mesh Subsample=2"

# Run the viewer
# --max-res 960: Higher than default (480p) but faster than 1280p
# --subsample 2: Creates a vertex for every 2nd pixel (good balance of density vs perf)
uv run da3d screen3d-viewer --max-res 960 --subsample 2
