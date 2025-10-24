# CLI Migration Instructions

The CLI code from the original `cli.py` (1500+ lines) needs to be migrated to the new package structure.

## Quick Migration Option

Due to the file size and complexity, here's the fastest way to get it working:

### Step 1: Copy Original CLI Temporarily

```bash
cd depth-anything-3d-viewer

# Copy the original cli.py as a starting point
cp ../cli.py da3d/cli/legacy.py
```

### Step 2: Create a Bridge Main File

Update `da3d/cli/main.py` to import from legacy temporarily:

```python
#!/usr/bin/env python3
import sys
import os

# Add parent Video-Depth-Anything to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(parent_dir, '..'))

# Import and run the original main
from da3d.cli.legacy import main as legacy_main

def main():
    legacy_main()

if __name__ == "__main__":
    main()
```

### Step 3: Update Imports in Legacy

Edit `da3d/cli/legacy.py` to use new package structure:

```python
# OLD imports:
from utils.projection_3d import DepthProjector, InteractiveParallaxController
from utils.viewer_3d import DepthMeshViewer, view_depth_3d, RealTime3DViewer

# NEW imports:
from da3d.projection import DepthProjector, InteractiveParallaxController
from da3d.viewing import DepthMeshViewer, view_depth_3d, RealTime3DViewer
```

### Step 4: Handle Video-Depth-Anything Imports

The original code imports:
```python
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream
from utils.dc_utils import read_video_frames, save_video
```

**Option A: Keep Original Repo in Path**
```bash
export PYTHONPATH="$PYTHONPATH:/path/to/Video-Depth-Anything"
```

**Option B: Copy Utils**
```bash
# Copy necessary utilities
cp ../utils/dc_utils.py da3d/utils/
```

### Step 5: Test

```bash
pip install -e .
da3d --help
da3d webcam3d
```

## Proper Migration (Long-term)

For a clean, modular structure, the CLI should be split as follows:

### File Structure
```
da3d/cli/
├── __init__.py          # Exports main()
├── main.py              # Argument parser, coordinates subcommands
├── config.py            # MODEL_CONFIGS, shared constants
├── utils.py             # Helper functions (save_recording, etc.)
├── webcam.py            # webcam_command, webcam3d_command
├── screen.py            # screen_command, screen3d_command, screen3d_viewer_command
├── viewer.py            # view3d_command
├── video.py             # video_command, process_video_batch/streaming
└── demo.py              # demo_command
```

### Migration Checklist

- [ ] Extract shared utilities to `utils.py`
- [ ] Move webcam commands to `webcam.py`
- [ ] Move screen commands to `screen.py`
- [ ] Move viewer commands to `viewer.py`
- [ ] Move video processing to `video.py`
- [ ] Move demo to `demo.py`
- [ ] Update all imports to use `da3d.*`
- [ ] Add register_commands() to each module
- [ ] Test each command individually

## Recommended Approach

**For immediate use:** Use the Quick Migration Option above (Steps 1-5)

**For clean codebase:** Gradually refactor legacy.py into proper modules

Would you like me to:
1. Create the quick migration files (get it working now)
2. Do the full modular split (takes longer but cleaner)
3. Provide a script to automate the migration
