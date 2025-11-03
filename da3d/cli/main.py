#!/usr/bin/env python3
"""
Main CLI entry point for Depth-Anything-3D viewer.
Currently uses legacy CLI from original codebase.
"""

import sys
import os

# Ensure parent Video-Depth-Anything is in path
# This assumes Video-Depth-Anything is in the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
video_depth_path = os.path.join(parent_dir, '..')
if os.path.exists(os.path.join(video_depth_path, 'video_depth_anything')):
    sys.path.insert(0, video_depth_path)

# Import main from legacy CLI
from da3d.cli.legacy import main as legacy_main


def main():
    """Entry point for da3d CLI."""
    # Handle 'status' command before importing dependencies
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        from da3d.cli.status import show_status
        show_status()
        return

    try:
        legacy_main()
    except ImportError as e:
        print("=" * 70)
        print("ERROR: Could not import Video-Depth-Anything dependencies")
        print("=" * 70)
        print(f"\nDetails: {e}\n")
        print("The Depth-Anything-3D package depends on the original")
        print("Video-Depth-Anything repository being available.")
        print("\nOptions to fix this:\n")
        print("1. Clone Video-Depth-Anything alongside this package:")
        print("   cd ..")
        print("   git clone https://github.com/DepthAnything/Video-Depth-Anything")
        print()
        print("2. Or add it to your PYTHONPATH:")
        print("   export PYTHONPATH=\"/path/to/Video-Depth-Anything:$PYTHONPATH\"")
        print()
        print("3. Or install it if/when it becomes pip-installable:")
        print("   pip install video-depth-anything")
        print()
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
