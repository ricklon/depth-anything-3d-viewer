# Feature Asset and Configuration Audit

This document tracks the assets and configuration required for each feature.

## ✅ Working Features (Assets Present)

### 1. **webcam3d** - Real-time Webcam 3D
- **Required Assets:** None (uses live camera)
- **Required Models:** `checkpoints/video_depth_anything_vits.pth` ✅ Present
- **Configuration:** `config.yaml` ✅ Present
- **Status:** ✅ Ready to test

### 2. **screen3d-viewer** - Screen Capture 3D
- **Required Assets:** None (captures live screen)
- **Required Models:** `checkpoints/video_depth_anything_vits.pth` ✅ Present
- **Required Dependencies:** `mss` library
- **Status:** ✅ Ready to test

### 3. **view3d** - Static 3D Viewing
- **Required Assets:**
  - Test image: `tests/data/test_image.jpg` ✅ Present
  - Test depth: `tests/data/test_depth.npy` ✅ Present (needs PNG conversion)
- **Status:** ⚠️ Needs depth PNG for easy testing

### 4. **Projector Commands** (projector-preview, projector-calibrate)
- **Configuration:** `config/projection_example.yaml` ✅ Present
- **Required Assets:**
  - `assets/test_pattern.png` ✅ Present
  - `assets/lobby_cube.obj` ✅ Present
  - `assets/art_rgb.jpg` ❌ **MISSING** (referenced in config)
  - `assets/art_depth.png` ❌ **MISSING** (referenced in config)
- **Status:** ⚠️ Config references missing assets

## ⚠️ Features Needing Attention

### 5. **X Key Capture** (High-Quality DA3)
- **Required Models:** Depth-Anything-3 checkpoint
- **Current Status:** ❌ No DA3 checkpoint found
- **Error Message:** "Warning: depth_anything_3 not found. DA3Estimator will fail to load."
- **Action Required:**
  - Document where to download DA3 checkpoint
  - Or disable/gracefully handle missing DA3 model
  - Or provide fallback to VDA model
- **Status:** ❌ Not functional

### 6. **Metric Depth Mode** (--metric flag)
- **Required Models:** `checkpoints/metric_video_depth_anything_vits.pth`
- **Current Status:** ❌ Metric checkpoint not found
- **Action Required:**
  - Document where to download metric model
  - Add to README download instructions
- **Status:** ❌ Not functional

### 7. **screen3d** - 2.5D Parallax
- **Required Assets:** None (captures live screen)
- **Required Dependencies:** `mss`, `pyvirtualcam` (optional)
- **Status:** ✅ Ready to test (minus virtual cam)

### 8. **video** - Process Video Files
- **Required Assets:** Example video file
- **Current Status:** ⚠️ No example video provided
- **Recommendation:** Add `tests/data/test_video.mp4`
- **Status:** ⚠️ No test asset

### 9. **webcam** - Basic Webcam (non-3D)
- **Required Assets:** None (uses live camera)
- **Status:** ✅ Ready to test

### 10. **demo** - Gradio Web Demo
- **Required Dependencies:** `gradio` library (optional)
- **Status:** ✅ Ready to test (if gradio installed)

### 11. **GUI Mode** (--gui flag)
- **Required Dependencies:** GUI viewer implementation
- **Status:** ✅ Ready to test

## 📋 Action Items

### High Priority
1. **Fix projection config** - Remove or provide missing art assets
2. **Document DA3 model** - Clarify X key capture requirements
3. **Document metric models** - Add download instructions
4. **Create test depth PNG** - Convert existing test_depth.npy to PNG

### Medium Priority
5. **Add example video** - For testing video processing
6. **Create examples directory** - Add Python API examples
7. **Add .gitattributes** - For LFS tracking of large assets

### Low Priority
8. **Document optional dependencies** - Clear guide for screen, virtual-cam, demo
9. **Add integration tests** - Automated testing for each command
10. **Create quickstart script** - One command to download all assets

## Asset Inventory

### Present Assets
```
checkpoints/
├── video_depth_anything_vits.pth    # 116 MB - Small model ✅
└── video_depth_anything_vitl.pth    # 1.5 GB - Large model ✅

assets/
├── test_pattern.png                 # 1.0 MB - Projector test ✅
└── lobby_cube.obj                   # 689 B - 3D scene ✅

tests/data/
├── test_image.jpg                   # 278 KB - Test image ✅
└── test_depth.npy                   # 3.7 MB - Depth array ✅

config/
└── projection_example.yaml          # Projector config ✅

config.yaml                          # Main config ✅
```

### Missing Assets
```
checkpoints/
├── metric_video_depth_anything_vits.pth    # Metric depth model ❌
├── depth_anything_3_*.pth                  # DA3 model for X key ❌

assets/
├── art_rgb.jpg                             # Referenced in config ❌
├── art_depth.png                           # Referenced in config ❌

tests/data/
├── test_depth.png                          # PNG version for docs ❌
└── test_video.mp4                          # Example video ❌

examples/
└── *.py                                    # Python API examples ❌
```

## Testing Matrix

| Feature | Assets Ready | Dependencies Ready | Tested | Notes |
|---------|--------------|-------------------|--------|-------|
| webcam3d | ✅ | ✅ | ⏳ | Needs VDA model |
| screen3d-viewer | ✅ | ⚠️ | ⏳ | Needs mss library |
| view3d | ⚠️ | ✅ | ⏳ | Needs depth PNG |
| screen3d | ✅ | ⚠️ | ⏳ | Needs mss library |
| video | ❌ | ✅ | ⏳ | Needs example video |
| webcam | ✅ | ✅ | ⏳ | Needs VDA model |
| demo | ✅ | ⚠️ | ⏳ | Needs gradio |
| projector-preview | ⚠️ | ✅ | ⏳ | Missing art assets |
| projector-calibrate | ✅ | ✅ | ⏳ | Can use test_pattern |
| X key (DA3) | ❌ | ❌ | ⏳ | Missing DA3 model |
| Metric depth | ❌ | ❌ | ⏳ | Missing metric model |
| GUI mode | ✅ | ✅ | ⏳ | Experimental |

## Recommendations

### Immediate Actions
1. **Fix projection config** to work with existing assets only
2. **Add download script** for optional models (DA3, metric)
3. **Convert test_depth.npy** to PNG for easier testing
4. **Document graceful degradation** when optional models missing

### Documentation Updates
- README: Add "Optional Models" section
- README: Clarify which features need which checkpoints
- Add DOWNLOAD_MODELS.md with complete model catalog
- Add TESTING.md guide for verifying each feature

### Code Improvements
- Add `--check` flag to verify all dependencies
- Improve error messages when models missing
- Add model download helper function
- Create setup wizard script
