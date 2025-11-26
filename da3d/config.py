"""
Configuration settings for Depth Anything 3D Viewer.
"""
from pathlib import Path
import os
import yaml

# --- Model Configurations ---
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# --- Load Config File ---
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / 'config.yaml'

config = {}
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load config.yaml: {e}")

# Helper to get nested config with default
def get_config(path, default):
    keys = path.split('.')
    val = config
    for key in keys:
        if isinstance(val, dict) and key in val:
            val = val[key]
        else:
            return default
    return val

# --- Default Paths ---
# Try to find checkpoints relative to the package or current directory
CHECKPOINTS_DIR = PROJECT_ROOT / get_config('model.checkpoints_dir', 'checkpoints')

if not CHECKPOINTS_DIR.exists():
    # Fallback to current directory if running from elsewhere
    CHECKPOINTS_DIR = Path('checkpoints')

# --- Camera Defaults ---
DEFAULT_FOCAL_LENGTH_X = get_config('camera.focal_length_x', 470.4)
DEFAULT_FOCAL_LENGTH_Y = get_config('camera.focal_length_y', 470.4)
DEFAULT_CAMERA_ID = get_config('camera.default_id', 0)

# --- Visualization Defaults ---
DEFAULT_MAX_RES = get_config('visualization.max_res', 640)
DEFAULT_INPUT_SIZE = get_config('visualization.input_size', 518)
DEFAULT_TARGET_FPS = get_config('visualization.target_fps', 30)

# --- High Quality Preset Defaults ---
HQ_MAX_RES = get_config('high_quality.max_res', 1024)
HQ_INPUT_SIZE = get_config('high_quality.input_size', 1022)
HQ_SUBSAMPLE = get_config('high_quality.subsample', 2)
HQ_SOR_NEIGHBORS = get_config('high_quality.sor_neighbors', 100)
HQ_SOR_STD_RATIO = get_config('high_quality.sor_std_ratio', 0.5)
HQ_METRIC_DEPTH_SCALE = get_config('high_quality.metric_depth_scale', 1.0)

# --- Default SOR Parameters ---
DEFAULT_SOR_NEIGHBORS = get_config('sor.neighbors', 50)
DEFAULT_SOR_STD_RATIO = get_config('sor.std_ratio', 1.0)
