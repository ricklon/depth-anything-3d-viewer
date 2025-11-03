#!/usr/bin/env python3
"""
Status checker for Depth-Anything-3D installation.
Verifies all dependencies and provides diagnostic information.
"""

import sys
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_status(name, status, details=""):
    """Print a status line with color."""
    if status:
        symbol = "[OK]"
        color = "\033[92m"  # Green
    else:
        symbol = "[--]"
        color = "\033[91m"  # Red
    reset = "\033[0m"

    print(f"{color}{symbol}{reset} {name}")
    if details:
        print(f"     {details}")


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    required = (3, 9)
    compatible = version >= required
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    return compatible, version_str


def check_module(module_name):
    """Check if a Python module is installed."""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        location = getattr(module, '__file__', 'unknown')
        return True, version, location
    except ImportError:
        return False, None, None


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"{device_name} (CUDA {cuda_version})"
        else:
            return False, "CUDA not available - CPU mode only"
    except Exception as e:
        return False, str(e)


def check_checkpoints():
    """Check for model checkpoints."""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return False, "checkpoints/ directory not found"

    models = {
        "video_depth_anything_vits.pth": "Small (97MB, fastest)",
        "video_depth_anything_vitb.pth": "Base (388MB, balanced)",
        "video_depth_anything_vitl.pth": "Large (1.3GB, best quality)",
    }

    found_models = []
    for model_file, description in models.items():
        model_path = checkpoints_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            found_models.append(f"{description}: {size_mb:.1f}MB")

    if found_models:
        return True, "\n  ".join(found_models)
    else:
        return False, "No model checkpoints found"


def show_status():
    """Display comprehensive installation status."""
    print_header("Depth-Anything-3D Installation Status")

    # Python version
    print("\n[Python Environment]")
    compatible, version = check_python_version()
    print_status(f"Python {version}", compatible,
                 "Requires Python 3.9+" if not compatible else "")

    # Core dependencies
    print("\n[Core Dependencies]")

    core_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("open3d", "Open3D"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("decord", "Decord"),
    ]

    for module, name in core_deps:
        installed, version, location = check_module(module)
        details = f"v{version}" if installed else "Not installed"
        print_status(name, installed, details)

    # Video-Depth-Anything
    print("\n[Video-Depth-Anything]")
    vda_installed, vda_version, vda_location = check_module("video_depth_anything")
    if vda_installed:
        if vda_location and vda_location != 'unknown':
            location_str = f"Installed at: {Path(vda_location).parent}"
        else:
            location_str = "Installed (location unknown)"
        print_status("Video-Depth-Anything", True, location_str)
    else:
        print_status("Video-Depth-Anything", False,
                     "Run: uv pip install -e ../Video-Depth-Anything")

    # Optional dependencies
    print("\n[Optional Dependencies]")

    optional_deps = [
        ("mss", "Screen capture support"),
        ("pyvirtualcam", "Virtual camera output"),
        ("gradio", "Web demo interface"),
        ("xformers", "Attention optimizations"),
    ]

    for module, description in optional_deps:
        installed, version, _ = check_module(module)
        details = f"v{version} - {description}" if installed else description
        print_status(module, installed, details)

    # GPU status
    print("\n[GPU Acceleration]")
    cuda_available, cuda_info = check_cuda()
    print_status("CUDA/GPU", cuda_available, cuda_info)

    # Model checkpoints
    print("\n[Model Checkpoints]")
    checkpoints_found, checkpoint_info = check_checkpoints()
    print_status("Checkpoints", checkpoints_found, checkpoint_info)

    # Installation status
    print_header("Summary")

    if vda_installed and cuda_available and checkpoints_found:
        print("\n\033[92m[OK] All systems ready!\033[0m")
        print("\nYou can now run:")
        print("  uv run da3d webcam3d")
        print("  uv run da3d screen3d-viewer")
    else:
        print("\n\033[93m[WARNING] Some components need attention:\033[0m")
        if not vda_installed:
            print("\n• Install Video-Depth-Anything:")
            print("  uv pip install -e ../Video-Depth-Anything")
        if not checkpoints_found:
            print("\n• Download model checkpoints:")
            print("  mkdir -p checkpoints && cd checkpoints")
            print("  curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth")
        if not cuda_available:
            print("\n• GPU acceleration disabled (CPU mode will be slow)")
            print("  See GPU_SETUP.md for installation instructions")

    print()


if __name__ == "__main__":
    show_status()
